#!/usr/bin/env bash
cd /Users/cptn/workbench/ai/colibri/validation/dsv4
./run_experiment.sh "N-gram speculation" spec_ngram.csv V4_DRAFT=4 V4_NGRAM=1
sleep 5
./run_experiment.sh "Full M T P speculation" spec_mtp.csv V4_MTP=1 V4_DRAFT=4 V4_NGRAM=0
sleep 5
./notify.sh "All speculation experiments complete. Ready for review"
