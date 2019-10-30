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

app.use(bodyParser.json({limit: '500mb'}));
app.use(bodyParser.raw({
  inflate: true,
  limit: '500mb',
  type: 'application/octet-stream'
}))
app.use(bodyParser.urlencoded({limit: '50mb', extended: true}));
app.set('view engine', 'pug')

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
    expires: new Date((new Date()).getTime() + ai.job_timeout).getTime() // aids.
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

app.get('/stats', async (req, res) => {
  try {
    // ...wow... kalorisparandet är högt här...
    await fs.copyFile("metrics.json", "metrics_read.json")
    await exec("R --no-save --no-restore < plot.r")
    const data = await fs.readFile("stats.png")
    return res.end(data, 'binary')
  }
  catch(err) {
    console.dir(err);
    return res.send(err)
  }
});

app.post('/newai', async (req, res) => {
  let {name, rollouts, jobs_per_generation, job_timeout, rom, script} = req.body
  if(!name || !rollouts || !job_timeout || !rom || !script) return res.sendStatus(500)
  if((name in ais) || !(rom in roms) || !(script in scripts)) return res.sendStatus(501)

  jobs_per_generation = jobs_per_generation || 5

  model = await createNewModel(name)
  ais[name] = {
    name,
    generation: 0,
    jobs_per_generation: parseInt(jobs_per_generation),
    jobs_done: 0,
    rollouts: parseInt(rollouts),
    job_timeout: parseInt(job_timeout),
    rom,
    model,
    script
  }
  await saveAI(ais[name])
  return res.sendStatus(200)
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
    + modelfile)
  const { stdout, stderr } = await exec('../bin/overmind update '
    + getModelFile(ai.name, ai.generation) + ' '
    + getExperienceFile(ai.name) + ' '
    + modelfile)

  console.log(stdout);

  fs.unlink(getExperienceFile(ai.name))
  console.log("going to delete files")
  files = await fs.readdir('./rollouts/');
  const rx = new RegExp('^' + ai.name + '\\.\\d+$')
  for(let i = 0; i < files.length; ++i) {
    console.log(files[i]);
    if(files[i].match(rx)) {
      fs.unlink('rollouts/' + files[i]);
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
}

app.post('/result/:job_id', async (req, res) => {
  const job_id = req.params.job_id
  const job = jobs[job_id]
  if(!job) return res.sendStatus(500)
  const ai = job.ai
  ++ai.jobs_done
  delete jobs[job_id]
  const experience = 'rollouts/' + ai.name + '.' + job_id
  console.log(`Job ${job_id} completed. Length ${req.body.length}`)
  await fs.writeFile(experience, req.body, 'binary')
  await fs.appendFile(getExperienceFile(ai.name), experience + '\n')
  if(ai.jobs_per_generation == ai.jobs_done) {
    await advanceGeneration(ai)
  } else {
    await saveAI(ai)
  }
  return res.sendStatus(200)
})


async function initialize() {
  const dirs = [ 'roms', 'scripts', 'ai', 'models', 'rollouts' ]
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

app.get('/', async (req, res) => {
  res.render('index', { title: 'Hey', message: 'Hello there!' })
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

app.listen(port, async () => {
  console.log(`Example app listening on port ${port}!`)
  await initialize()
})

