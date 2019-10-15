const express = require('express')
const app = express()
const bodyParser = require('body-parser');
const util = require('util');
const exec = util.promisify(require('child_process').exec);
const fs = require('fs').promises
const port = 3000

const job_id = 0
const ais = []
const scripts = []
const roms = []
const models = []

app.use(bodyParser.json());
app.set('view engine', 'pug')

async function createNewModel() {
  await exec('../bin/overmind -create model')
  model_data = await fs.readFile('model', 'binary')
  models.push(model_data)
  return models.length - 1
}

function createJob(ai) {
  job_id += 1
  const j = {
    job_id,
    expired: new Date() + ai.job_timeout,
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

function createAI({name, rollouts, job_timeout, rom, script}) {
  ais.push({
    name,
    jobs: {},
    rollouts_left: rollouts,
    rollouts,
    job_timeout,
    rom_id: createROM(rom),
    model_id: createNewModel(),
    script_id: createScript(script)
  })
  return ais.length - 1
}

app.post('/newai', (req, res) => {
  console.log(req.body);
  res.send({ ai_id: createAI(req.body)})
})

app.get('/job', (req, res) => {
  const ai = ais[req.body.ai_id]
  if(!ai.rollouts_left) {
    return res.sendStatus(541)
  }
  --ai.rollouts_left
  return res.send(createJob(ai))
})

app.post('/result', (req, res) => {

})

app.get('/', function (req, res) {
  res.render('index', { title: 'Hey', message: 'Hello there!' })
})

app.listen(port, () => console.log(`Example app listening on port ${port}!`))