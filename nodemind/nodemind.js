const express = require('express')
const app = express()
const bodyParser = require('body-parser');
const util = require('util');
const exec = util.promisify(require('child_process').exec)
const fs = require('fs').promises
const port = 3000

let job_id = 0
const ais = []
const scripts = []
const roms = []
const models = []

app.use(bodyParser.json({limit: '50mb'}));
app.use(bodyParser.raw({
  inflate: true,
  limit: '50mb',
  type: 'application/octet-stream'
}))
app.use(bodyParser.urlencoded({limit: '50mb', extended: true}));
app.set('view engine', 'pug')

async function createNewModel() {
  await exec('../bin/overmind create model')
  model_data = await fs.readFile('model', 'binary')
  console.log(model_data.length)
  models.push(model_data)
  return models.length - 1
}

function createJob(ai) {
  job_id += 1
  const j = {
    job_id,
    expires: new Date((new Date()).getTime() + parseInt(ai.job_timeout)).getTime(),
    model_id: ai.model_id,
    rom_id: ai.rom_id,
    script_id: ai.script_id
  }
  ai.jobs[job_id] = j
  return j
}

function createScript(s) {
  scripts.push(s)
  return scripts.length - 1
}

function createROM(r) {
  roms.push(r)
  return roms.length - 1
}

async function createAI({name, rollouts, job_timeout, rom, script}) {
  ais.push({
    name,
    jobs: {},
    rollouts_left: rollouts,
    rollouts,
    job_timeout,
    rom_id: createROM(rom),
    model_id: await createNewModel(),
    script_id: createScript(script)
  })
  return ais.length - 1
}

app.post('/newai', async (req, res) => {
  console.log(req.body);
  res.send({ ai_id: await createAI(req.body)})
})

app.get('/rom/:id', (req, res) => {
  const rom = roms[req.params.id]
  console.log(req.params)
  if(!rom) return res.sendStatus(500)
  return res.end(rom, 'binary')
})

app.get('/script/:id', (req, res) => {
  const script = scripts[req.params.id]
  if(!script) return res.sendStatus(500)
  return res.send(script)
})

app.get('/model/:id', (req, res) => {
  const model = models[req.params.id]
  if(!model) return res.sendStatus(500)
  return res.end(model, 'binary')
})

app.get('/job/:aiid', (req, res) => {
  const ai = ais[req.params.aiid]
  if(!ai) {
    return res.sendStatus(500)
  }
  if(!ai.rollouts_left) {
    return res.sendStatus(541)
  }
  --ai.rollouts_left
  return res.send(createJob(ai))
})

app.post('/result/:aiid/:jobid', async (req, res) => {
  const ai = ais[req.params.aiid]
  if(!ai) return res.sendStatus(500)
  if(!(req.params.jobid in ai.jobs)) return res.sendStatus(501)
  delete ai.jobs[req.params.jobid]
  const name = req.params.aiid + "_" + req.params.jobid;
  await fs.writeFile(name, req.body, 'binary')
  await fs.appendFile(req.params.aiid + '.result', name + '\n')


  return res.sendStatus(200)
})

app.get('/', function (req, res) {
  res.render('index', { title: 'Hey', message: 'Hello there!' })
})

app.listen(port, () => console.log(`Example app listening on port ${port}!`))

