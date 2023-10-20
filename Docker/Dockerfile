#Basic layer> Ubuntu
FROM ubuntu:jammy
MAINTAINER giuliano.iorio.astro@gmail.com

ENV PATH="/root/miniconda3/bin:$PATH"
ENV PYTHONPATH="${PYTHONPATH}:/ca23/software/tsunami-newspin/build/python"
ENV DIRPATH="/ca23/"
WORKDIR $DIRPATH
ARG TARGETPLATFORM
ARG SOFTWAREDIR=$DIRPATH/software

#1 layer: install g++, git, cmake
RUN  apt update && apt -y dist-upgrade \
&& apt update \
&& apt install -y   \
g++ \
make \
cmake \
git \
wget \
sudo \
vim \
nano \
emacs \
unzip \
libgsl-dev


#RUN groupadd --gid $USER_GID $USERNAME \
#    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -p $USERNAME \
#    && echo $USERNAME all=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
#    && chmod 0440 /etc/sudoers.d/$USERNAME

#2 layer: miniconda \
RUN case "${TARGETPLATFORM}" in \
         "linux/amd64")  export MINICONDAS=Miniconda3-py310_23.5.2-0-Linux-x86_64.sh ;; \
         "linux/arm64")  export MINICONDAS=Miniconda3-py310_23.5.2-0-Linux-aarch64.sh  ;; \
    esac; \
     cd /root \
     && wget    https://repo.anaconda.com/miniconda/${MINICONDAS} --no-check-certificate \
     &&  bash ${MINICONDAS}  -b \
     && rm -f ${MINICONDAS}


#3 create env and basic python packages \
RUN  pip install numpy pandas matplotlib scipy scikit-learn jupyter cython swig

#4 layer: create software_dir
RUN mkdir $SOFTWAREDIR

#5 layer: tsunami
ADD tsunami-newspin.zip /ca23/software/
RUN cd $SOFTWAREDIR && unzip tsunami-newspin.zip \
    && rm -rf tsunami-newspin.zip \
    && cd tsunami-newspin/build && cmake .. -Dpython=TRUE \
    && make

#6 layer: SEVN \
# clone and build SEVN fork,
# the variable SEVN_VER is used to breack the Docker cache and alreay rebuilt this layer so
# that we install always the last version.
# It is rebuilt everytime the SEVN_VER change, so to rebuild it everytime it is enough
# just to use a timestamp, e.g. --build-arg SEVN_VER=$(date +%Y%m%d-%H%M%S)
RUN  cd $SOFTWAREDIR && git clone https://gitlab.com/sevncodes/sevn.git \
&& cd sevn && mkdir build &&  cd build && cmake .. && make \
&& cd  ../.. && cp sevn/run_scripts/run.sh .


#7 Install SEVNpy
RUN cd $SOFTWAREDIR/sevn/SEVNpy && pip install .

#8 PyFalcon
RUN  cd $SOFTWAREDIR && git clone https://gitlab.com/iogiul/pyfalcon.git \
     && cd pyfalcon &&  pip install .

#9 PyTorch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

#10 Keras, Tensorflow
RUN pip install keras tensorflow

#11 Cython gsl
RUN cd $SOFTWAREDIR && git clone https://github.com/twiecki/CythonGSL.git \
    && cd CythonGSL && pip install .

#12 Mystuff: roteasy
RUN cd $SOFTWAREDIR && git clone https://github.com/giulianoiorio/roteasy.git \
    && cd roteasy && pip install .

#13 Mystuff: FERMI, pyblack
RUN pip install fermi pyblack colossus

#14 Mystuff: galpynamics
RUN cd $SOFTWAREDIR && git clone https://gitlab.com/iogiul/galpynamics \
    && cd galpynamics && pip install .

#15 Mystuff: OpOpGadget
RUN cd $SOFTWAREDIR && git clone https://github.com/giulianoiorio/OpOpGadget.git \
    && cd OpOpGadget && pip install .

#16 Install sphynx for documentation \
RUN  pip install  sphinx sphinx-rtd-theme

#17 Install pytest \
RUN  pip install  pytest

#18 Install twine \
RUN  pip install  twine

#Final: Setup easy jupyter access
RUN mkdir -p $HOME/.jupyter/ \
    && echo "c.NotebookApp.token=''" >> $HOME/.jupyter/jupyter_notebook_config.py;

#Clean to reduce
RUN apt-get clean && pip cache purge


CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

EXPOSE 8888