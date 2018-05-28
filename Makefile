# This Makefile wraps commands used to set up the learning environment.

setup:
	# Set PYTHONPATH 
	export PYTHONPATH=`pwd`
	
	# Set up nlp tools
	tar xf nlp_tools/spellcheck/most_common.tar.xz --directory nlp_tools/spellcheck/
	
	# Install Python packages
	pip install -r requirements.txt
