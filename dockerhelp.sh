DOCKER_GPU_PARAMS=""
DOCKER_COMMAND="nvidia-docker"

if ! [ -x "$(command -v nvidia-docker)" ]; then
	DOCKER_COMMAND="docker"
	DOCKER_GPU_PARAMS="--gpus=all"
fi

if ! [ -x "$(command -v nvidia-smi)" ]; then
	DOCKER_GPU_PARAMS=""
fi

run_docker() {
echo $DOCKER_COMMAND $DOCKER_GPU_PARAMS
$DOCKER_COMMAND run \
	$DOCKER_GPU_PARAMS \
	-u $(id -u):$(id -g) \
	--network=host \
	-e XAUTHORITY \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	--volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
	--volume="$HOME/.nv/:/thehome/.nv/:rw" \
	--env="DISPLAY" \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	-e "TERM=xterm-256color" \
	-e "RESUME=$RESUME" \
	-e "AIS=$AIS" \
	-e "JOBS=$JOBS" \
	-e "LR=$LR" \
	-e "OMP_NUM_THREADS=1" \
	-e "GPU=$GPU" \
	--rm -it \
	-v $(pwd):/nesai \
	-e HOME=/thehome \
	-v $HOME/.bash_history:/thehome/.bash_history \
	-w /nesai/ \
	--cap-add=SYS_PTRACE \
	--security-opt seccomp=unconfined \
	$1 bash -c "$2"
}
