import requests
import sys
import json
import os
import subprocess

def download_to(url, path):
	r = requests.get('%s' % (url))
	if 200 != r.status_code:
		raise Exception('Unable to download: %s' % (path))
	open(path, 'wb').write(r.content)

server = os.getenv('aiserver') or 'http://localhost:3000'

if len(sys.argv) < 2:
	print('%s <ai_name>' % (sys.argv[1]))
	sys.exit(1)

rom_path = '/tmp/inspect.nes'
script_path = '/tmp/inspect.lua'
model_path = '/tmp/inspect.model'

while True:
	r = requests.get('%s/ai/%s' % (server, sys.argv[1]))
	if 200 != r.status_code:
		raise Exception('Bad response code when reading AI: %d' % (r.status_code))

	ai = json.loads(r.text)

	print('Inspecting "%s" on generation %d' % (ai['name'], ai['generation']))

	download_to('%s/rom/%s' % (server, ai['rom']), rom_path)
	download_to('%s/script/%s' % (server, ai['script']), script_path)
	download_to('%s/model/%s' % (server, ai['model']), model_path)

	env = os.environ.copy()
	env['MODEL'] = model_path
	env['HUMAN'] = "0"
	env['BE'] = script_path
	env['ROLLOUTS'] = "1"
	p = subprocess.Popen(['./hqn_quicknes', rom_path], cwd='bin/', stdout=None, env=env)
	p.wait()


