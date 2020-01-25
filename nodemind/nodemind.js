const express = require('express')
const app = express()
const bodyParser = require('body-parser');
const util = require('util');
const exec = util.promisify(require('child_process').exec)
const spawn = require('child_process').spawn
const fs = require('fs').promises
const crypto = require('crypto')

const port = 3000

let job_id = 0
const ais = {}
const scripts = {}
const roms = {}
const models = {}
const jobs = {}
const shouldPlotAIs = []

app.use(bodyParser.json({limit: '5000mb'}));
app.use(bodyParser.raw({
  inflate: true,
  limit: '5000mb',
  type: 'application/octet-stream'
}))
app.use(bodyParser.urlencoded({limit: '5000mb', extended: true}));
app.set('view engine', 'pug')
app.use(express.static('public'))

function getModelFile(name, generation) {
  return 'models/' + name + '.' + generation + '.model'
}

function getExperienceFile(name) {
  return 'rollouts/' + name + '.result'
}

async function createNewModel(name) {
  const filename = getModelFile(name, 0)
  await exec('../bin/overmind create ' + filename)
  const data = await fs.readFile(filename)
  models[name] = { data }
  return name
}

function createJob(ai) {
  job_id += 1
  jobs[job_id] = {
    ai,
    expires: new Date((new Date()).getTime() + (ai.rollouts * 100000)).getTime() // aids.
  }
  return {
    job_id,
    rollouts: ai.rollouts,
    ai: ai.name,
    rom: ai.rom,
    script: ai.script
  }
}

async function saveAI(ai) {
  await fs.writeFile('ai/' + ai.name + '.json', JSON.stringify(ai))
}

/* Plots value network reward estimate vs observed rewards */
app.get('/valuestats/:name', async (req, res) => {
  const n = parseInt(req.params.nGenerations)
  try {
    //await fs.copyFile("metrics.json", "metrics_read.json")
    const name = req.params.name
    if(!name || !(name in ais)) return res.sendStatus(500)

    return res.render('stats', {
      title: `Value Stats - ${name}`,
      ais,
      file: `/valuestats_${name}.png`
    })
  }
  catch(err) {
    console.dir(err);
    return res.send(err)
  }
});

/* Plots the full statistics but for the nGenerations last generations */
app.get('/smallstats/:name/:nGenerations', async (req, res) => {
  const n = parseInt(req.params.nGenerations)
  try {
    const name = req.params.name
    if(!name || !(name in ais)) return res.sendStatus(500)
    return res.render('stats', {
      title: `Small Stats - ${name}`,
      ais,
      file: `/smallstats_${name}.png`
    })
  }
  catch(err) {
    console.dir(err);
    return res.send(err)
  }
});

async function generateValuestats(ai)
{
    var name = ai.name;
    let { stdout } = await exec(`ls -1t metrics/${name}_*.json | head -1`)
    const files = stdout.split('\n').join(' ')
    {
    let cmd = `jq -s . ${files} > metrics_read_value_${name}.json`
    let { stdout } = await exec(cmd)
    }
    {
    let cmd = "R --no-save --no-restore < plot_val.r"
    let { stdout } = await exec(cmd,
    {
      env: {
        PLOT_DATA_FILE: `metrics_read_value_${name}.json`,
        PLOT_FILE: `valstats_${name}.png`
      }
    })
    }
    const data = await fs.readFile(`valstats_${name}.png`)
    await fs.writeFile(`public/valuestats_${name}.png`, data)
}

async function generateSmallstats(ai, n)
{
    var name = ai.name;
    let { stdout } = await exec(`ls -1t metrics/${name}_*.json | head -${n}`)
    const files = stdout.split('\n').join(' ')
    {
    let cmd = `jq -s . ${files} > metrics_read_small_${name}.json`
    let { stdout } = await exec(cmd)
    }
    {
    let cmd = "R --no-save --no-restore < plot_stats.r"
    let { stdout } = await exec(cmd,
    {
      env: {
        PLOT_DATA_FILE: `metrics_read_small_${name}.json`,
        PLOT_FILE: `smallstats_${name}.png`
      }
    })
    }
    var out_path = `smallstats_${name}.png`;
    const data = await fs.readFile(`smallstats_${name}.png`)
    await fs.writeFile("public/" + out_path, data)
    return {
      path: "public/" + out_path,
      url: "/" + out_path
    }
}

async function generateLargestats(ai)
{
    var name = ai.name;
    await exec(`jq -s . metrics/${name}_*.json > metrics_read_large_${name}.json`)
    await exec("R --no-save --no-restore < plot_stats.r",
    {
      env: {
        PLOT_DATA_FILE: `metrics_read_large_${name}.json`,
        PLOT_FILE: `stats_${name}.png`
      }
    })
    const data = await fs.readFile(`stats_${name}.png`)
    var out_path = `largestats_${name}.png`;
    await fs.writeFile("public/" + out_path, data)
    return {
      path: "public/" + out_path,
      url: "/" + out_path
    }
}

async function generateRewards(ai)
{
  var name = ai.name;
  await exec(`jq -s . metrics/${name}_*.json > metrics_read_reward_${name}.json`)
  await exec("R --no-save --no-restore < plot_rewards.r",
  {
    env: {
      PLOT_DATA_FILE: `metrics_read_reward_${name}.json`,
      PLOT_FILE: `rewards_${name}.png`
    }
  })
  const data = await fs.readFile(`rewards_${name}.png`)
  await fs.writeFile(`public/rewards_${name}.png`, data)
}

app.get('/largestats/:name', async (req, res) => {
  try {
    const name = req.params.name
    if(!name || !(name in ais)) return res.sendStatus(500)
    //await fs.copyFile("metrics.json", "metrics_read.json")
    return res.render('stats', {
      title: `Large Stats - ${name}`,
      ais,
      file: `/largestats_${name}.png`
    })
  }
  catch(err) {
    console.dir(err);
    return res.send(err)
  }
});

app.get('/rewards/:name', async (req, res) => {
  try {
    const name = req.params.name
    if(!name || !(name in ais)) return res.sendStatus(500)
    return res.render('stats', {
      title: `Rewards - ${name}`,
      ais,
      file: `/rewards_${name}.png`
    })
  }
  catch(err) {
    console.dir(err);
    return res.send(err)
  }
});

app.post('/newai', async (req, res) => {
  let {name, rollouts, jobs_per_generation, rom, script} = req.body
  if(!name || !rollouts || !rom || !script) return res.sendStatus(500)
  if((name in ais) || !(rom in roms) || !(script in scripts)) return res.sendStatus(501)

  jobs_per_generation = jobs_per_generation || 5

  model = await createNewModel(name)
  ais[name] = {
    name,
    generation: 0,
    jobs_per_generation: parseInt(jobs_per_generation),
    jobs_done: 0,
    rollouts: parseInt(rollouts),
    rom,
    model,
    script
  }
  await saveAI(ais[name])
  return res.sendStatus(200)
})

app.get('/ai/:name', async (req, res) => {
  const name = req.params.name
  if(!name || !(name in ais)) return res.sendStatus(500)
  return res.send(ais[name])
})

app.get('/generation/:name', (req, res) => {
  const ai = ais[req.params.name]
  if(!ai) return res.sendStatus(500)
  return res.send({ generation: ai.generation })
})

app.get('/roms', (req, res) => {
  let ret = []
  for(let k in roms) {
    ret.push(k)
  }
  return res.send(ret)
})

app.get('/scripts', (req, res) => {
  let ret = []
  for(let k in scripts) {
    ret.push(k)
  }
  return res.send(ret)
})

app.get('/rom/:name', (req, res) => {
  const name = req.params.name
  if(!(name in roms)) return res.sendStatus(500)
  return res.end(roms[name], 'binary')
})

app.get('/script/:name', (req, res) => {
  const name = req.params.name
  if(!(name in scripts)) return res.sendStatus(500)
  return res.send(scripts[name])
})

app.get('/model/:name', (req, res) => {
  const name = req.params.name
  if(!(name in models)) return res.sendStatus(500)
  return res.end(models[name].data, 'binary')
})

function pendingJobs(name) {
  let pending = 0
  for(let key in jobs) {
    if(jobs[key].ai.name == name) {
      pending++
    }
  }
  return pending
}

async function generateMetrics(ai)
{
try {
    const name = req.params.name;
    await exec(`jq -s . metrics/${name}_*.json > metrics_read_${name}.json`)
  }
  catch(err) {
    console.dir(err);
    return res.send(err)
  }
}

async function generateGif(ai) {
  const modelFile = getModelFile(ai.name, ai.generation);
  const cmd = '../bin/hqn_quicknes ' + 'roms/' + ai.rom;
  const gif = 'gifs/' + ai.name + "_" + ("00000" + ai.generation).slice(-5) + '.gif';
  let env = {
    'HUMAN': '0',
    'MODEL': modelFile,
    'BE': 'scripts/' + ai.script,
    'ROLLOUTS': "1",
    'HL': "1",
    'FPS': "0",
    'EXPFILE': "",
    'GIF': gif,
    'MAX_FRAMES': 3000,
    'RESET': true
  }
  env = Object.assign(env, process.env);
  // console.dir(env)
  try {
    const { stdout, stderr } = await exec(cmd, {env: env});
    fs.rename(gif, "public/" + gif);
    // console.log(stdout)
  }
  catch(e)
  {
    console.dir(e);
  }
  // console.log(stderr)
  // const data = await fs.readFile(gif);
  // return res.end(data, 'binary')
}


app.get('/job/:name', (req, res) => {
  const ai = ais[req.params.name]
  if(!ai) return res.sendStatus(500)
  if((ai.jobs_done + pendingJobs(ai.name)) >= ai.jobs_per_generation) {
    return res.sendStatus(541)
  }
  return res.send(createJob(ai))
})


async function advanceGeneration(ai) {
  const modelfile = getModelFile(ai.name, ai.generation + 1)
  console.log('../bin/overmind update '
    + getModelFile(ai.name, ai.generation) + ' '
    + getExperienceFile(ai.name) + ' '
    + modelfile + ' '
    + ai.generation + ' '
    + ai.name)
    try {
  const { stdout, stderr } = await exec('../bin/overmind update '
    + getModelFile(ai.name, ai.generation) + ' '
    + getExperienceFile(ai.name) + ' '
    + modelfile + ' '
    + ai.generation + ' '
    + ai.name)

  // console.log(stdout);
    }
    catch(e)
    {
      console.dir(e);
    }

  fs.unlink(getExperienceFile(ai.name))
  // console.log("going to delete files")
  files = await fs.readdir('./rollouts/');
  const rx = new RegExp('^' + ai.name + '\\.\\d+$')
  var saved = false;
  for(let i = 0; i < files.length; ++i) {
    // console.log(files[i]);
    if(files[i].match(rx)) {
      if(!saved)
      {
        await fs.copyFile('rollouts/' + files[i], 'saved_rollouts/' + files[i].split(".")[0] + '.g' + ai.generation);
        await fs.unlink('rollouts/' + files[i]);
        saved = true;
      }
      else
      {
        await fs.unlink('rollouts/' + files[i]);
      }
    }
  }
  // Set new model as active
  delete models[ai.name]
  const data = await fs.readFile(modelfile)
  models[ai.name] = { data }
  // Reset AI and save it for next generation
  ai.jobs_done = 0
  ++ai.generation
  await saveAI(ai)
  shouldPlotAIs.push(ai.name);
}

app.post('/result/:job_id', async (req, res) => {
  const job_id = req.params.job_id
  const job = jobs[job_id]
  if(!job) {
    console.log(`Completed job ${job_id} does not exist (probably timed out) :(`)
    return res.sendStatus(500)
  }
  const ai = job.ai
  const completed = ++ai.jobs_done
  delete jobs[job_id]
  const experience = 'rollouts/' + ai.name + '.' + job_id
  console.log(`Job ${job_id} completed. Length ${req.body.length}. Total done ${ai.jobs_done}`)
  if(!req.query.local) {
    await fs.writeFile(experience, req.body, 'binary')
  }
  await fs.appendFile(getExperienceFile(ai.name), experience + '\n')
  if(ai.jobs_per_generation == completed) {
    await advanceGeneration(ai)
  } else {
    await saveAI(ai)
  }
  return res.sendStatus(200)
})


async function initialize() {
  const dirs = [ 'roms', 'scripts', 'ai', 'models', 'rollouts', 'public', 'public/gifs', 'saved_rollouts']
  for(d of dirs) {
    try { await fs.mkdir(d) } catch(e) {}
  }
  
  rom_files = await fs.readdir('roms')
  for(name of rom_files) {
    console.log('Loading ROM', name)
    const data = await fs.readFile('roms/' + name)
    roms[name] = data
  }

  script_files = await fs.readdir('scripts')
  for(name of script_files) {
    console.log('Loading script', name)
    scripts[name] = await fs.readFile('scripts/' + name)
  }

  ai_files = await fs.readdir('ai')
  for(name of ai_files) {
    console.log('Loading AI', name)
    const ai = JSON.parse(await fs.readFile('ai/' + name))
    const data = await fs.readFile(getModelFile(ai.name, ai.generation))
    models[ai.name] = { data }
    ais[ai.name] = ai
    if(ai.jobs_done >= ai.jobs_per_generation) {
      console.log('AI ' + name + ' was interrupted in a generation transition. Will advance...')
      await advanceGeneration(ai)
    }
  }
}

app.get('/gifz', async (req, res) => {
  var gifs = await fs.readdir('public/gifs')
  gifs.sort()
  res.render('gifz', { gifs: gifs})
})

app.get('/gifz/:name', async (req, res) => {
  const ai = ais[req.params.name]
  if(!ai) return res.sendStatus(500)
  var gifs = await fs.readdir('public/gifs')
  gifs.sort()
  const rx = new RegExp(ai.name + '_\\d+\\.gif')
  gifs = gifs.filter(a => null != a.match(rx))
  for(let i = 0; i < gifs.length; ++i) { 
    gifs[i] = '/gifs/' + gifs[i]
  }
  // console.log(gifs)
  res.render('gifz', { ais, gifs })
})

async function getGifs() {
  var gifs = await fs.readdir('public/gifs')
  gifs.sort()
  var ret = {}
  for(let ai in ais) {
    const rx = new RegExp(ais[ai].name + '_\\d+\\.gif')
    ai_gifs = gifs.filter(a => null != a.match(rx))
    for(let i = 0; i < ai_gifs.length; ++i) { 
      ai_gifs[i] = '/gifs/' + ai_gifs[i]
    }
    ret[ai] = ai_gifs
    ret[ai].reverse()
  }
  return ret
}

app.get('/', async (req, res) => {
  var gifs = await getGifs();
  res.render('dash', { ais, gifs })
})

setInterval(() => {
  const now = (new Date()).getTime()
  let nuke = []
  for(let key in jobs) {
    if(now >= jobs[key].expires) {
      nuke.push(key)
    }
  }
  for(let n of nuke) {
    delete jobs[n] // tarded.
  }
  if(nuke.length) console.log('Number of jobs timed out', nuke.length)
}, 100);

async function generatePlots() {
  if(shouldPlotAIs.length > 0)
  {
    var name = shouldPlotAIs.pop();
    console.log("Will generate plots for " + name)
    await Promise.all([
      generateGif(ais[name]),
      generateLargestats(ais[name]),
      generateSmallstats(ais[name], 5),
      generateValuestats(ais[name]),
      generateRewards(ais[name])
    ]);
    console.log("Plots for " + name + " done!");
    generatePlots();
    return
  }
  console.log("No more plots to do...");
  await new Promise((resolve) => { setTimeout(resolve, 1000)});
  generatePlots();
}

app.listen(port, async () => {
  console.log(`Example app listening on port ${port}!`)
  await initialize()
  generatePlots();
})

