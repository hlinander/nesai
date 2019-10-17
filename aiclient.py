import requests
import os
import sys
import time
import json

model = None
rom = None
script = None

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

def download_model(m):
	global model
	global name
	if model == m:
		return
	open('/tmp/%s.model' % (name), 'wb').write(download('/model/%s' % (m)))
	model = m

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

def run_job(j):
	download_model(j['model'])
	download_rom(j['rom'])
	download_script(j['script'])

print('Using AI server: %s' % (server))
print('Will work on "%s" as "%s".' % (ai, name))

while True:
	try:
		r = get('/job/%s' % (ai))
		if 200 == r.status_code:
			run_job(json.loads(r.text))
			break
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

