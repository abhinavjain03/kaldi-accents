#!/bin/bash

# d is as c, but with one extra layer.

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

# note: the last column is a version of tdnn_d that was done after the
# changes for the 5.1 version of Kaldi (variable minibatch-sizes, etc.)
# System                  tdnn_c   tdnn_d       tdnn_d[repeat]
# WER on train_dev(tg)      17.37     16.72      16.51
# WER on train_dev(fg)      15.94     15.31      15.34
# WER on eval2000(tg)        20.0      19.2        19.2
# WER on eval2000(fg)        18.2      17.8       17.7
# Final train prob       -1.43781  -1.22859      -1.22215
# Final valid prob       -1.56895    -1.354     -1.31647

stage=0


scale=false
scale_factor=-1
scale_file=""
affix=
train_stage=-10
has_fisher=false
speed_perturb=true
common_egs_dir=
reporting_email=
remove_egs=true
nj=25


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi
dir=exp/nnet3/tdnn_d
dir=$dir${affix:+_$affix}
dir=${dir}$suffix
train_set=cv_train_nz$suffix
ali_dir=exp/tri4_cv_train_nz_ali$suffix

local/nnet3/run_ivector_common.sh --stage $stage \
        --speed-perturb $speed_perturb || exit 1;






if [ $stage -le 9 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $ali_dir/tree | grep num-pdfs | awk '{print $2}')

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 dim=1024
  relu-renorm-layer name=tdnn2 input=Append(-1,2) dim=1024
  relu-renorm-layer name=tdnn3 input=Append(-3,3) dim=1024
  relu-renorm-layer name=tdnn4 input=Append(-3,3) dim=1024
  relu-renorm-layer name=tdnn5 input=Append(-7,2) dim=1024
  relu-renorm-layer name=tdnn6 dim=1024

  output-layer name=output input=tdnn6 dim=$num_targets max-change=1.5 presoftmax-scale-file=$dir/configs/presoftmax_prior_scale.vec


EOF

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 10 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi


  steps/nnet3/train_dnn_added_scaling.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs 2 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.optimization.initial-effective-lrate 0.0017 \
    --trainer.optimization.final-effective-lrate 0.00017 \
    --egs.dir "$common_egs_dir" \
    --scale $scale \
    --scale-factor $scale_factor \
    --scale-file $scale_file \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 100 \
    --use-gpu true \
    --feat-dir=data/${train_set}_hires \
    --ali-dir $ali_dir \
    --lang data/lang \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;

fi

graph_dir=exp/tri4/graph_sw1_tg
if [ $stage -le 11 ]; then

  mfccdev=1
  calculatehiresdev=1
  ivectordev=1
  decodedev=1
  devsets="cv_test_onlyindian cv_dev_nz cv_test_onlynz"

  mfccdir=exp/mfcc
  if [ $mfccdev -eq 1 ]; then
    for x in $devsets; do
        steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_16k.conf --cmd "$train_cmd" \
                           data/$x exp/make_mfcc/$x $mfccdir
        steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
        utils/fix_data_dir.sh data/$x
    done
  fi

  #calculate hires mfccs
  mfcchiresdir=exp/mfcc_hires
  if [ $calculatehiresdev -eq 1 ]; then
    for x in $devsets; do
      utils/copy_data_dir.sh data/$x data/${x}_hires
      steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires_16k.conf \
            --cmd "$train_cmd" data/${x}_hires exp/make_mfcc/${x}_hires $mfcchiresdir;
        steps/compute_cmvn_stats.sh data/${x}_hires exp/make_mfcc/${x}_hires $mfcchiresdir;

      utils/fix_data_dir.sh data/${x}_hires
    done

    echo "============================================="
    echo "HIRES MFCCs DONE!!"
    echo "============================================="
  fi

  
  if [ $ivectordev -eq 1 ]; then
    for x in $devsets; do
      online_ivector_dir=exp/nnet3/ivectors_${x}
            steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
        data/$x  exp/nnet3/extractor ${online_ivector_dir} || exit 1;

        #       online_ivector_dir=exp/nnet3/ivectors_${x}_hires
        #     steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
        # data/${x}_hires  exp/alreadyTrainedModelsOnSwbd/nnet_online/extractor ${online_ivector_dir} || exit 1;



        echo "============================================="
        echo "CALCULATION OF IVECTORS DONE!!!!"
        echo "============================================="
    done
  fi

  if [ $decodedev -eq 1 ]; then
    for decode_set in $devsets; do
      (
      num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" \
        --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
        $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}_hires_sw1_tg || exit 1;
      ) &
    done
  fi
fi
wait;
exit 0;
