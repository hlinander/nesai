from threading import Thread, Lock

import requests
import os
import sys
import time
import json
import subprocess

mutex = Lock()

generation = None
rom = None
script = None
generation = None
result_queue = []

server = os.getenv('aiserver') or 'http://localhost:3000'
ai = sys.argv[1]
name = sys.argv[2]

if len(sys.argv) < 3:
	print('Usage: <ai name> <client name>')
	sys.exit(1)

def get(path):
	global server
	return requests.get('%s%s' % (server, path))

def download(path):
	r = get(path)
	if 200 != r.status_code:
		raise Exception('Unable to download: %s' % (path))
	return r.content

def upload(path, ul):
	global server
	r = requests.post('%s%s' % (server, path), data=ul, headers = { 'Content-Type': 'application/octet-stream' })
	if 200 != r.status_code:
		raise Exception('Unable to upload: %s' % (path))

def download_model(ai):
	global generation
	global name
	r = get('/generation/' + ai)
	if 200 != r.status_code:
		raise Exception('Cant read ai generation')	
	print(r.text)
	curr_gen = int(json.loads(r.text)['generation'])
	if generation == curr_gen: 
		return
	try:
		os.unlink('/tmp/%s.model' % (name))
	except:
		pass	
	open('/tmp/%s.model' % (name), 'wb').write(download('/model/%s' % (ai)))
	generation = curr_gen

def download_rom(dl):
	global rom
	global name
	if rom == dl:
		return
	open('/tmp/%s.nes' % (name), 'wb').write(download('/rom/%s' % (dl)))
	rom = dl

def download_script(s):
	global script
	global name
	if script == s:
		return
	open('/tmp/%s.lua' % (name), 'wb').write(download('/script/%s' % (s)))
	script = s

def results_thread():
	global result_queue
	while True:
		r = []
		mutex.acquire()
		r = result_queue
		result_queue = []
		mutex.release()
		if 0 == len(r):
			time.sleep(0.5)
			continue
		for it in r:
			try:
				if not os.path.exists(it['result_file']):
					raise Exception('#### ASSOCIATED RESULT FILE IS MISSING FIX (job_id: %d).' % (job_id))
				upload('/result/%s' % (it['job_id']), open(it['result_file'], 'rb').read())
				os.unlink(it['result_file'])
			except e as Exception:
				print('RESULT_THREAD: %s' % (e))
				continue
			print('Successfully uploaded job: %s' % (it['job_id']))

def result_queue_happy():
	mutex.acquire()
	n = len(result_queue)
	mutex.release()
	return (n < 32)

experience_id = 0

def run_job(j):
	global result_queue
	global experience_id

	download_model(j['ai'])
	download_rom(j['rom'])
	download_script(j['script'])

	env = os.environ.copy()
	model_path = f"/tmp/{name}.model"
	env['MODEL'] = model_path
	env['BE'] = f'/tmp/{name}.lua'
	p = subprocess.Popen(['./hqn_quicknes', f'/tmp/{name}.nes'], cwd='bin/', stdout=None, env=env)
	p.wait()
	experience_id += 1 # hack fuck
	old_experience = "%s.experience" % (model_path)
	new_experience = "%s.%d" % (old_experience, experience_id)
	os.rename(old_experience, new_experience)
	if 0 == p.returncode:
		mutex.acquire()
		result_queue.append({ 'job_id': j['job_id'], 'result_file': new_experience })
		mutex.release()
	else:
		print('Sad client %d' % (p.returncode))


print('Using AI server: %s' % (server))
print('Will work on "%s" as "%s".' % (ai, name))

t = Thread(target=results_thread)
t.start()

while True:
	try:
		while not result_queue_happy():
			print('Too many pending results in queue (upload too slow?)')
			time.sleep(1.0)
		r = get('/job/%s' % (ai))
		if 200 == r.status_code:
			run_job(json.loads(r.text))
		elif 541 == r.status_code:
			print('No jobs... trying again in 1 second.')
			time.sleep(1.0)
		else:
			print('Server failed with %d (%s).' % (r.status_code, r.text))
			break
	except Exception as e:
		print('EXCEPTION: %s' % (e))
		print('Sleeping one second before trying to resume...')
		time.sleep(1.0)


