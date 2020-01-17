from threading import Thread, Lock

import requests
import os
import sys
import time
import json
import subprocess
import traceback

mutex = Lock()

generation = None
rom = None
script = None
generation = None
result_queue = []
result_size = 0

server = os.getenv('aiserver') or 'http://localhost:3000'
ai = sys.argv[1]
name = sys.argv[2]

if len(sys.argv) < 3:
	print('Usage: <ai name> <client name>')
	sys.exit(1)

def is_local():
	return (server[:len('http://localhost:')] == 'http://localhost:') or (server[:len('http://127.0.0.1:')] == 'http://127.0.0.1:')

def get(path):
	global server
	return requests.get('%s%s' % (server, path))

def download(path):
	r = get(path)
	if 200 != r.status_code:
		raise Exception('Unable to download: %s (code: %d)' % (path, r.status_code))
	return r.content

def post(path):
	global server
	return requests.post('%s%s' % (server, path))

def upload(path, ul):
	global server
	r = requests.post('%s%s' % (server, path), data=ul, headers = { 'Content-Type': 'application/octet-stream' })
	if 200 != r.status_code:
		raise Exception('Unable to upload: %s (code: %d)' % (path, r.status_code))

def get_model_name(name):
	if is_local():
		return '%s.model' % (name)
	return '/tmp/%s.model' % (name)

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
		os.unlink(get_model_name(name))
	except:
		pass	
	open(get_model_name(name), 'wb').write(download('/model/%s' % (ai)))
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

def get_result():
	global result_queue
	r = None
	mutex.acquire()
	if 0 != len(result_queue):
		r = result_queue[0]
		result_queue = result_queue[1:]
	mutex.release()
	return r

def results_thread():
	global result_queue
	global result_size
	while True:
		r = get_result()
		if None == r:
			time.sleep(0.5)
			continue
		try:
			if not os.path.exists(r['result_file']):
				raise Exception('#### ASSOCIATED RESULT FILE IS MISSING FIX (job_id: %d).' % (job_id))
			upload('/result/%s' % (r['job_id']), open(r['result_file'], 'rb').read())
			os.unlink(r['result_file'])
			print('Successfully uploaded job: %s (%d bytes)' % (r['job_id'], r['size']))
		except Exception as e:
			print('### RESULT_THREAD EXCEPTION: %s' % (e))
		mutex.acquire()
		result_size -= r['size']
		print('Backlog remaining: %f mb' % (result_size / (1024*1024)))
		mutex.release()

def backlog_within_reason():
	if is_local():
		return True
	mutex.acquire()
	n = result_size
	mutex.release()
	return (n < (500 * 1024 * 1024))

experience_id = 0

def run_job(j):
	global result_queue
	global result_size
	global experience_id

	download_model(j['ai'])
	download_rom(j['rom'])
	download_script(j['script'])

	env = os.environ.copy()
	model_path = get_model_name(name)
	env['MODEL'] = model_path if not is_local() else '../%s.model' % (name) # VERY GOOD DONT TOUCH
	env['BE'] = f'/tmp/{name}.lua'
	env['ROLLOUTS'] = str(j['rollouts'])
	p = subprocess.Popen(['./hqn_quicknes', f'/tmp/{name}.nes'], cwd='bin/', stdout=None, env=env)
	p.wait()
	
	if 0 == p.returncode:
		experience_id += 1 # hack fuck
		old_experience = "%s.experience" % (model_path)
		if is_local():
			os.rename(old_experience, 'nodemind/rollouts/%s.%d' % (j['ai'], j['job_id']))
			post('/result/%d?local=hampus' % (j['job_id']))
		else:
			new_experience = "%s.%d" % (old_experience, experience_id)

			os.rename(old_experience, new_experience)
			filesize = os.path.getsize(new_experience)

			mutex.acquire()
			result_size += filesize
			result_queue.append({
				'job_id': j['job_id'],
				'result_file': new_experience,
				'size': filesize
			})
			print('queue: %d, size: %d' % (len(result_queue), filesize))
			mutex.release()
	else:
		print('Sad client %d' % (p.returncode))


print('Using AI server: %s' % (server))
print('Will work on "%s" as "%s".' % (ai, name))

if not is_local():
	print('I AM RUNNING WITH REMOTE UPLOAD')
	t = Thread(target=results_thread)
	t.start()
else:
	print('I AM RUNNING WITH LOCAL UPLOAD')

while True:
	try:
		while not backlog_within_reason():
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
		traceback.print_exc()
		print('Sleeping one second before trying to resume...')
		time.sleep(1.0)


