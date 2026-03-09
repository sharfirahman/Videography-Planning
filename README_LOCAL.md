##This readme is for running the code without the docker

#Before you start, you can remove the folder "docker-scripts" and the "DockerFile"

#Install and update the necessary ubuntu tools
sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y install tar
sudo apt-get -y install wget

#Install python3 (preferebly 3.10-3.12)
sudo apt-get -y install python3-pip

#Install pip and all the other specific libraries
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir "numpy<2.0"
pip install --no-cache-dir "matplotlib==3.5.2"
pip install --no-cache-dir tikzplotlib
pip install --no-cache-dir notebook

Download the specific julia version in the project folder:
        - wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.2-linux-x86_64.tar.gz 
        - tar -xzf julia-1.8.2-linux-x86_64.tar.gz
    - Set environment variables properly
    - In order to not alter the original path, create a ssymbolic link for the julia path setup which is handled by conda:
            ln -sf $HOME/dev/MultiDroneMultiActorFilming/julia-1.8.2/bin/julia $CONDA_PREFIX/bin/julia
            export JULIA_LOAD_PATH="$PROJECT_DIR:$PROJECT_DIR/src/mdma_greedy:$PROJECT_DIR/env"
            export JULIA_NUM_THREADS=12
            export DISPLAY="$DISPLAY"
    - If you are using conda, set the variable in conda environment so conda can handle it:
        conda env config vars set PROJECT_DIR="$HOME/dev/MultiDroneMultiActorFilming"
        conda env config vars set JULIA_LOAD_PATH="$PROJECT_DIR:$PROJECT_DIR/src/mdma_greedy:$PROJECT_DIR/env"
        conda env config vars set JULIA_NUM_THREADS=12
        conda env config vars set DISPLAY="$DISPLAY"

- Create a new folder in the project folder, and copy the juliapackages into the file. Run julia command to activate and instantiate the package.

        - create a directory /env : mkdir -p env
		- copy the files in the /juliapackages: cp -r juliapackages/* env/
		- go the env folder: cd env
		- Install the julia packages : julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate();'

If you want to use Blender, install the blender in the same project folder.Also, create a symbolic link for the blender path
    - create a directory /blender : mkdir -p blender (might be already there from the previous version)
    - go the env folder: cd blender
    - download and install the specific version: 
        - wget https://mirror.clarkson.edu/blender/release/Blender4.2/blender-4.2.1-linux-x64.tar.xz
        - tar xvf /airlab/blender/blender-4.2.1-linux-x64.tar.xz
    - Create symbolic link:
        ln -sf $HOME/dev/MultiDroneMultiActorFilming/blender/blender-4.2.1-linux-x64/blender $CONDA_PREFIX/bin/blender

Install other dependancies(use sudo if needed):
apt-get -y install build-essential git subversion cmake libx11-dev libxxf86vm-dev libxcursor-dev libxi-dev libxrandr-dev libxinerama-dev libegl-dev libxrender-dev libsm-dev
apt-get -y install libwayland-dev wayland-protocols libxkbcommon-dev libdbus-1-dev linux-libc-dev


How to run the code:(From the main README):

* `julia`
* `julia> using MDMA`

To run all the experiments and generate all outputs
* `julia> conf = ExperimentsConfig("./experiments")`
* `julia> run_all_experiments(conf)`

ExperimentsConfig can also be provided a list of experiments to run (in the case you do not want to run everything).
For example, to run only the `cluster` experiment you can do
* `julia> conf = ExperimentsConfig("./experiments", ["cluster"])`

The repo comes with a set of solutions already in the correct locations. You can also try
to render all the image outputs, and compute solution evaluations without recomputing solutions.
* `julia> blender_render_all_experiments(conf)`
* `julia> evaluate_all_experiments(conf)`


