#!/bin/bash

set -e

lang_res=data/language_res
text_res=data/text_classification_res
chinese_ner_res=data/chinese_ner_res
seq_label_res=data/sequence_labeling_res
lang_model=~/language_model/fluid
lang_data=~/NLP_icode/NLP/language/test_set/fake_realtitle.test
chinese_ner_model=~/NLP_icode/NLP/chinese_ner/chinese_ner/model/paddle_for_anakin/params_batch_1920000/
chinese_ner_data=~/NLP_icode/NLP/chinese_ner/chinese_ner/test_set/data_file
seq_label_model=~/NLP_icode/NLP/chinese_ner/sequence_labeling/model/params_batch_450000/
seq_label_data=~/NLP_icode/NLP/chinese_ner/sequence_labeling/test_set/perf-eval.legoraw
text_model=~/NLP_icode/NLP/text_classification/model/bilstm_model/
text_data=~/NLP_icode/NLP/text_classification/test_set/out.ids.txt

mkdir -p data

unset KMP_AFFINITY
export KMP_AFFINITY="granularity=fine,compact,0,0" # when HT if OFF
# export KMP_AFFINITY="granularity=fine,compact,1,0" # when HT is ON

# 1 socket for 8180
# echo 0 > /proc/sys/kernel/numa_balancing
core_num=`nproc`
threads_per_core=`lscpu | grep "per core" | awk -F ':' '{print $2}' | sed 's/^ *\| *$//g'`
sockets=`lscpu | grep -i "socket(s)" | awk -F ':' '{print $2}' | sed 's/^ *\| *$//g'`
thread_per_socket=`expr $threads_per_core \* $sockets`
core_num=`expr $core_num / $thread_per_socket`
echo ${core_num}

core_idx=`expr ${core_num} - 1`
core_range='0-'${core_idx}

echo ${core_range}

#unset OMP_NUM_THREADS
#export OMP_NUM_THREADS=${core_num}
#unset MKL_NUM_THREADS
#export MKL_NUM_THREADS=${core_num}

run_exec=./build/paddle/fluid/inference/tests/book/test_inference_nlp

cat /proc/cpuinfo |grep name |head -n 1 > $lang_res
git checkout ./paddle/fluid/inference/tests/book/test_inference_nlp.cc
cd build
make -j
cd ..
taskset -c ${core_range} numactl -l  $run_exec -model_path $lang_model -data_file $lang_data -num_threads 12 > $lang_res\_12.txt  2>&1
taskset -c ${core_range} numactl -l  $run_exec -model_path $lang_model -data_file $lang_data -num_threads 10 > $lang_res\_10.txt 2>&1
taskset -c ${core_range} numactl -l  $run_exec -model_path $lang_model -data_file $lang_data -num_threads 6 > $lang_res\_6.txt 2>&1
taskset -c ${core_range} numactl -l  $run_exec -model_path $lang_model -data_file $lang_data -num_threads 2 > $lang_res\_2.txt 2>&1
taskset -c ${core_range} numactl -l  $run_exec -model_path $lang_model -data_file $lang_data -num_threads 1 > $lang_res\_1.txt 2>&1


cat /proc/cpuinfo |grep name |head -n 1 > $text_res
git checkout ./paddle/fluid/inference/tests/book/test_inference_nlp.cc
cd build
make -j
cd ..
$taskset -c ${core_range} numactl -l run_exec -model_path $text_model -data_file $text_data -num_threads 1 > $text_res\_1.txt 2>&1
$taskset -c ${core_range} numactl -l run_exec -model_path $text_model -data_file $text_data -num_threads 2 > $text_res\_2.txt 2>&1
$taskset -c ${core_range} numactl -l run_exec -model_path $text_model -data_file $text_data -num_threads 6 > $text_res\_6.txt 2>&1
$taskset -c ${core_range} numactl -l run_exec -model_path $text_model -data_file $text_data -num_threads 10 > $text_res\_10.txt 2>&1
$taskset -c ${core_range} numactl -l run_exec -model_path $text_model -data_file $text_data -num_threads 12 > $text_res\_12.txt 2>&1

cat /proc/cpuinfo |grep name |head -n 1 > $chinese_res
cp ./paddle/fluid/inference/tests/book/test_inference_nlp.cc.chinese_ner ./paddle/fluid/inference/tests/book/test_inference_nlp.cc
cd build
make -j
cd ..
$taskset -c ${core_range} numactl -l run_exec -model_path $chinese_ner_model -data_file $chinese_ner_data -num_threads 1 > $chinese_ner_res\_1.txt 2>&1
$taskset -c ${core_range} numactl -l run_exec -model_path $chinese_ner_model -data_file $chinese_ner_data -num_threads 2 > $chinese_ner_res\_2.txt 2>&1
$taskset -c ${core_range} numactl -l run_exec -model_path $chinese_ner_model -data_file $chinese_ner_data -num_threads 6 > $chinese_ner_res\_6.txt 2>&1
$taskset -c ${core_range} numactl -l run_exec -model_path $chinese_ner_model -data_file $chinese_ner_data -num_threads 10 > $chinese_ner_res\_10.txt 2>&1
$taskset -c ${core_range} numactl -l run_exec -model_path $chinese_ner_model -data_file $chinese_ner_data -num_threads 12 > $chinese_ner_res\_12.txt 2>&1

cat /proc/cpuinfo |grep name |head -n 1 > $seq_label_res
cp ./paddle/fluid/inference/tests/book/test_inference_nlp.cc.sequence_labeling ./paddle/fluid/inference/tests/book/test_inference_nlp.cc
cd build
make -j
cd ..
$taskset -c ${core_range} numactl -l run_exec -model_path $seq_label_model -data_file $seq_label_data -num_threads 1 > $seq_label_res\_1.txt 2>&1
$taskset -c ${core_range} numactl -l run_exec -model_path $seq_label_model -data_file $seq_label_data -num_threads 2 > $seq_label_res\_2.txt 2>&1
$taskset -c ${core_range} numactl -l run_exec -model_path $seq_label_model -data_file $seq_label_data -num_threads 6 > $seq_label_res\_6.txt 2>&1
$taskset -c ${core_range} numactl -l run_exec -model_path $seq_label_model -data_file $seq_label_data -num_threads 10 > $seq_label_res\_10.txt 2>&1
$taskset -c ${core_range} numactl -l run_exec -model_path $seq_label_model -data_file $seq_label_data -num_threads 12 > $seq_label_res\_12.txt 2>&1
