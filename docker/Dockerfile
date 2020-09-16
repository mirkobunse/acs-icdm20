FROM continuumio/miniconda3

# declare the build-time variables to be set by the Makefile
ARG group
ARG gid
ARG user
ARG uid

# install utilities and create the user
RUN apt-get update --fix-missing && \
    apt-get install -y \
      nano \
      build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid $gid $group && \
    useradd  --gid $gid --uid $uid $user

# install the environment
ADD .environment.yml .bashrc /home/$user/
RUN conda install --channel anaconda setuptools && \
    conda update --name base --channel defaults conda && \
    conda env create --file /home/$user/.environment.yml && \
    chown --recursive $user:$group /home/$user

# Before the environment is created, conda is updated to suppress
# later occuring deprecation warnings. However, this update needs
# setuptools to be re-installed.
# See: https://superuser.com/a/1456350

WORKDIR /mnt/home
USER $uid
RUN conda init bash && \
    echo "conda activate acs-sandbox" >> /home/$user/.bashrc

