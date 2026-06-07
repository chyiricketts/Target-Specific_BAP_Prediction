#!/bin/bash
#PBS -l select=1:ncpus=1:mem=16gb
#PBS -l walltime=10:00:00
#PBS -N out_trainingsets

cd $PBS_O_WORKDIR

source ~/miniforge3/etc/profile.d/conda.sh
conda activate wip

echo "Started at $(date)"

QUERY="/rds/general/user/cr725/home/aev-plig_research/fep/fep_benchmark_inputs/structure_inputs/jacs_set/mcl1_extra_flips_protein.pdb"
INPUT="../all_proteins.csv"
OUT="tm_mcl1_results1.csv"

echo "unique_id,tm_score" > "$OUT"

tail -n +2 "$INPUT" | while IFS=, read -r id path
do
    if [ ! -f "$path" ]; then
        echo "$id,NA" >> "$OUT"
        continue
    fi

    result=$(TMalign "$QUERY" "$path")

    tm=$(echo "$result" | awk '/TM-score=/{for(i=1;i<=NF;i++) if($i=="TM-score="){print $(i+1); exit}}')

    if [ -z "$tm" ]; then
        tm="NA"
    fi

    echo "$id,$tm" >> "$OUT"
done

echo "Finished at $(date)"