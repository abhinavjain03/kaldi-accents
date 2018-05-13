. ./cmd.sh
. ./path.sh

nj=20
modelDirectory=/exp/abhinav/accents_exp
trainSet=cv_train_nz
bnfNnetModelDir=/exp/abhinav/accent_recognizer_exp/nnet3/tdnn_1024nodes_300bnlayer_except_newzealand+adaptonlynz
affix=
train_stage=-10
bnf_dim=1024

. utils/parse_options.sh


dir=exp/nnet3/tdnn_${affix}


mfcc=0
mfcchires=0
alignData=0
getAccentAligns=0
get_bnf_features=1
append_bnf_mfcc=1
config=1
egsTrain=1
train=1
decode=1
wer=1



mfccdir=exp/mfcc
if [ $mfcc -eq 1 ]; then
	for x in cv_train_all_with_accents cv_dev_all_with_accents; do
		steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_16k.conf --cmd "$train_cmd" \
				data/$x exp/make_mfcc/$x $mfccdir
		steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
		utils/fix_data_dir.sh data/$x
	done
fi

mfccdir=exp/mfcc_hires
if [ $mfcchires -eq 1 ]; then
  for x in cv_train_all_with_accents cv_dev_all_with_accents; do
    utils/copy_data_dir.sh data/$x data/${x}_hires
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires_16k.conf --cmd "$train_cmd" \
        data/${x}_hires exp/make_hires/$x $mfccdir;
    steps/compute_cmvn_stats.sh data/${x}_hires exp/make_hires/${x} $mfccdir;
    utils/fix_data_dir.sh data/${x}_hires
  done
fi

dataalidir=exp/tri4_${trainSet}_sp_ali
if [ $alignData -eq 1 ]; then

    steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
             data/${trainSet}_sp data/lang $modelDirectory/tri4 $dataalidir

fi

##After aligning the data merge all the alignments into one file,
##and generate accent posteriors
accentalidir=exp/tri4_train_with_accents_ali
if [ $getAccentAligns -eq 1 ]; then

    mkdir test
    mkdir $accentalidir
    for id in $(seq $nj); do gunzip -c $dataalidir/ali.$id.gz; done | ali-to-pdf $dataalidir/final.mdl ark:- ark,t:test/ali.scp
    export LD_LIBRARY_PATH=/usr/lib/atlas-base:/exp/sw/kaldi/src/lib:/exp/sw/kaldi/tools/openfst-1.6.2/src/lib/.libs:$LD_LIBRARY_PATH
    ./temp data/$trainSet/accent_num test/ali.scp "ark:|gzip -c > $accentalidir/ali.1.gz"
    echo "1" > $accentalidir/num_jobs
    cp $dataalidir/final.mdl $accentalidir
    cp $dataalidir/tree $accentalidir

fi

bnf_feat_dir=data/${trainSet}_sp_bnf
if [ $get_bnf_features -eq 1 ]; then

    steps/nnet3/make_bottleneck_features.sh \
    --nj $nj \
    --use-gpu true \
    --cmd "$train_cmd" \
        tdnn_bn.renorm data/${trainSet}_sp_hires data/${trainSet}_sp_bnf \
        $bnfNnetModelDir exp/make_bnf/${trainSet}_sp exp/make_bnf


fi

appended_dir=data/${trainSet}_mfcc_bnf_appended_sp
dump_bnf_dir=exp/append_mfcc_bnf
if [ $append_bnf_mfcc -eq 1 ]; then

    steps/append_feats.sh \
        --cmd "$train_cmd" \
        --nj $nj \
      $bnf_feat_dir data/${trainSet}_sp_hires $appended_dir \
      exp/append_hires_mfcc_bnf/${trainSet}_sp $dump_bnf_dir || exit 1;
    steps/compute_cmvn_stats.sh $appended_dir \
        exp/make_cmvn_mfcc_bnf $dump_bnf_dir || exit 1;

fi




if [ $config -eq 1 ]; then

  mkdir -p $dir/configs

  num_pdfs=`tree-info ${dataalidir}/tree 2>/dev/null | grep num-pdfs | awk '{print $2}'` || exit 1;
  feat_dim=`feat-to-dim scp:${appended_dir}/feats.scp -`


  cat <<EOF > $dir/configs/network.xconfig
  input dim=$feat_dim name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 input=Append(-2,-1,0,1,2) dim=1024
  relu-renorm-layer name=tdnn2 dim=1024
  relu-renorm-layer name=tdnn3 input=Append(-1,2) dim=1024
  relu-renorm-layer name=tdnn4 input=Append(-3,3) dim=1024
  relu-renorm-layer name=tdnn5 input=Append(-3,3) dim=1024
  relu-renorm-layer name=tdnn6 input=Append(-7,2) dim=1024
  relu-renorm-layer name=tdnn_bn dim=$bnf_dim

  relu-renorm-layer name=prefinal-affine-1 input=tdnn_bn dim=1024
  output-layer name=output dim=${num_pdfs} max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/

fi



if [ $egsTrain -eq 1 ]; then
  cmd=run.pl
  left_context=16
  right_context=12

  context_opts="--left-context=$left_context --right-context=$right_context"

    transform_dir=${dataalidir}
    cmvn_opts="--norm-means=false --norm-vars=false"
    extra_opts=()
    extra_opts+=(--cmvn-opts "$cmvn_opts")
    extra_opts+=(--left-context $left_context)
    extra_opts+=(--right-context $right_context)
    echo "$0: calling get_egs.sh for generating examples with alignments as output"


  steps/nnet3/get_egs.sh $egs_opts "${extra_opts[@]}" \
    --num-utts-subset 300 \
    --nj $nj \
      --samples-per-iter 400000 \
      --cmd "$cmd" \
      --frames-per-eg 8 \
      $appended_dir ${dataalidir} $dir/egs || exit 1;

fi



if [ $train -eq 1 ]; then

  steps/nnet3/train_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs 2 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 9 \
    --trainer.optimization.initial-effective-lrate 0.0017 \
    --trainer.optimization.final-effective-lrate 0.00017 \
    --egs.dir $dir/egs \
    --cleanup.preserve-model-interval 20 \
    --use-gpu true \
    --ali-dir $dataalidir \
    --lang data/lang \
    --feat-dir=$appended_dir \
    --reporting.email="$reporting_email" \
    --dir $dir  || exit 1;

fi

graph_dir=exp/tri4/graph_sw1_tg
if [ $decode -eq 1 ]; then

  mfccdev=0
  calculatehiresdev=0
  ivectordev=0
  get_bnf_features_dev=1
  append_bnf_mfcc=1
  decodedev=1
  # devsets="cv_test_onlyindian cv_dev_nz cv_test_onlynz"
  devsets="cv_test_onlynz"

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
  
  if [ $get_bnf_features_dev -eq 1 ]; then

    for x in $devsets; do
      bnf_feat_dir=data/${x}_bnf
      steps/nnet3/make_bottleneck_features.sh \
      --nj $nj \
      --use-gpu true \
      --cmd "$train_cmd" \
          tdnn_bn.renorm data/${x}_hires $bnf_feat_dir \
          $bnfNnetModelDir exp/make_bnf/${x} exp/make_bnf
    done

  fi

  
  dump_bnf_dir=exp/append_mfcc_bnf
  if [ $append_bnf_mfcc -eq 1 ]; then

    for x in $devsets; do 
      bnf_feat_dir=data/${x}_bnf
      appended_dir=data/${x}_mfcc_bnf_appended
      
      steps/append_feats.sh \
          --cmd "$train_cmd" \
          --nj $nj \
        $bnf_feat_dir data/${x}_hires $appended_dir \
        exp/append_hires_mfcc_bnf/${x} $dump_bnf_dir || exit 1;
      steps/compute_cmvn_stats.sh $appended_dir \
          exp/make_cmvn_mfcc_bnf $dump_bnf_dir || exit 1;
    done

  fi



  if [ $decodedev -eq 1 ]; then
    for decode_set in $devsets; do
      appended_dir=data/${decode_set}_mfcc_bnf_appended
      num_jobs=`cat $appended_dir/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" \
        $graph_dir $appended_dir $dir/decode_${decode_set}_mfcc_bnf_appended || exit 1;
      
    done
  fi
fi


if [ $wer -eq 1 ]; then
  #for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 
  for x in ${dir}/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
  #for x in exp/*/*/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
fi
