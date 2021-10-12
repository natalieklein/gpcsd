#!/bin/bash

if [ ! -d auditory_lfp/data ]; then
    mkdir -p auditory_lfp/data
fi
if [ ! -d neuropixels/data ]; then
    mkdir -p neuropixels/data
fi


# Auditory data
if [ ! -f auditory_lfp/data/time.txt ]; then
    wget https://zenodo.org/record/5137888/files/time.txt -P auditory_lfp/data/
fi      
if [ ! -f auditory_lfp/data/medial_evoked_mua.txt ]; then
    wget https://zenodo.org/record/5137888/files/medial_evoked_mua.txt -P auditory_lfp/data/
fi
if [ ! -f auditory_lfp/data/lateral_evoked_mua.txt ]; then
    wget https://zenodo.org/record/5137888/files/lateral_evoked_mua.txt -P auditory_lfp/data/
fi

for i in {1..24}
do
    if [ ! -f auditory_lfp/data/lateral_electrode$i.txt ]; then
        wget https://zenodo.org/record/5137888/files/lateral_electrode$i.txt -P auditory_lfp/data/
    fi
    if [ ! -f auditory_lfp/data/medial_electrode$i.txt ]; then
        wget https://zenodo.org/record/5137888/files/medial_electrode$i.txt -P auditory_lfp/data/
    fi
done

# Neuropixels data
if [ ! -f neuropixels/data/mouse405751.lfp.nwb ]; then
    wget https://zenodo.org/record/5150708/files/mouse405751.lfp.nwb -P neuropixels/data/
fi
if [ ! -f neuropixels/data/mouse405751.spikes.nwb ]; then
    wget https://zenodo.org/record/5150708/files/mouse405751.spikes.nwb -P neuropixels/data/
fi