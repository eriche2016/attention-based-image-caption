#!/bin/bash
# will change to different directory for different datasets
cd flickr8k-caption
python myeval.py $1
cd ../
