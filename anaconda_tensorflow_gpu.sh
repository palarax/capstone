#!/usr/bin/env bash

env_name=$1

if [ -z $env_name ];
then
	echo "Please provide new environment name"
	exit
fi

#https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc
conda create --name $env_name tensorflow-gpu 
