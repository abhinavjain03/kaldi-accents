. ./cmd.sh
. ./path.sh

# ./test --num-pdfs=9054 --num-accents=16 --frame-subsampling-factor=1 --online-ivectors=scp:exp/nnet3/ivectors_cv_train_nz_sp/ivector_online.scp --online-ivector-period=10 --left-context=16 --right-context=12 --compress=true --num-frames=8 "ark,s,cs:utils/filter_scp.pl --exclude exp/nnet3/tdnn_d_sp/egs/valid_uttlist data/cv_train_nz_sp_hires/split6/1/feats.scp | apply-cmvn --norm-means=false --norm-vars=false --utt2spk=ark:data/cv_train_nz_sp_hires/split6/1/utt2spk scp:data/cv_train_nz_sp_hires/split6/1/cmvn.scp scp:- ark:- |" "ark,s,cs:filter_scp.pl data/cv_train_nz_sp_hires/split6/1/utt2spk exp/nnet3/tdnn_d_sp/egs/ali.scp | ali-to-pdf exp/tri4_cv_train_nz_ali_sp/final.mdl scp:- ark:- | ali-to-post ark:- ark:- |" scp:cv-train-accents ark:test123.ark


#newName=finetuning_cslu_6accents_
newName=
dir=exp/nnet3/tdnn_d_${newName}sp

#for egs
num_utts_subset=512 #default is 512
samples_per_iter=400000 #default is 400000



mkdir $dir
initialDir=exp/nnet3/tdnn_d_sp
model=$initialDir/final.mdl



train_stage=3
train_set=train_cslu_6accents
nj=20

initialize=0
calculateMfcc=0
alignments=0
calculateHires=0
ivector=0
egs=0
train=0
decode=0
wer=1


if [ $initialize -eq 1 ]; then

nnet3-am-copy --edits="set-learning-rate name=input learning-rate=0;
						set-learning-rate name=lda learning-rate=0;
						set-learning-rate name=ivector learning-rate=0;
						set-learning-rate name=tdnn1.affine learning-rate=0;
						set-learning-rate name=tdnn1.relu learning-rate=0;
						set-learning-rate name=tdnn1.renorm learning-rate=0;
						set-learning-rate name=tdnn2.affine learning-rate=0;
						set-learning-rate name=tdnn2.relu learning-rate=0;
						set-learning-rate name=tdnn2.renorm learning-rate=0;
						set-learning-rate name=tdnn3.affine learning-rate=0;
						set-learning-rate name=tdnn3.relu learning-rate=0;
						set-learning-rate name=tdnn3.renorm learning-rate=0;
						set-learning-rate name=tdnn4..affine learning-rate=0;
						set-learning-rate name=tdnn4.relu learning-rate=0;
						set-learning-rate name=tdnn4.renorm learning-rate=0;
						set-learning-rate name=tdnn5.affine learning-rate=0;
						set-learning-rate name=tdnn5.relu learning-rate=0;
						set-learning-rate name=tdnn5.renorm learning-rate=0;" \
						$model $dir/3.mdl

	# nnet3-am-copy $model $dir/3.mdl

	echo "============================================="
	echo "MODEL IS COPIED WITH NEW LEARNING RATES!!!"
	echo "============================================="
fi




#calculate mfccs for ivectors
mfccdir=exp/mfcc
if [ $calculateMfcc -eq 1 ]; then
	utils/fix_data_dir.sh data/${train_set}
	steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_16k.conf \
        --cmd "$train_cmd" data/${train_set} exp/make_mfcc/${train_set} $mfccdir;
    steps/compute_cmvn_stats.sh data/${train_set} exp/make_mfcc/${train_set} $mfccdir;

	utils/fix_data_dir.sh $data/${train_set}
	echo "============================================="
	echo "STANDARD MFCCs DONE!!!"
	echo "============================================="

fi

#get alignments
ali_dir=exp/tri4_${train_set}_ali
if [ $alignments -eq 1 ]; then

	steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/${train_set} data/lang exp/tri4 ${ali_dir}
    echo "============================================="
    echo "ALIGNMENT OF DATA DONE!!!"
    echo "============================================="
fi


#calculate hires mfccs
mfcchiresdir=exp/mfcc_hires
if [ $calculateHires -eq 1 ]; then
	utils/copy_data_dir.sh data/${train_set} data/${train_set}_hires
	steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires_16k.conf \
        --cmd "$train_cmd" data/${train_set}_hires exp/make_mfcc/${train_set}_hires $mfcchiresdir;
    steps/compute_cmvn_stats.sh data/${train_set}_hires exp/make_mfcc/${train_set}_hires $mfcchiresdir;

	utils/fix_data_dir.sh data/${train_set}_hires

	echo "============================================="
	echo "HIRES MFCCs DONE!!"
	echo "============================================="
fi

#find ivectors
online_ivector_dir=exp/nnet3/ivectors_${train_set}
if [ $ivector -eq 1 ]; then

	  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
	    data/${train_set}  exp/nnet3/extractor ${online_ivector_dir} || exit 1;
	    echo "============================================="
	    echo "CALCULATION OF IVECTORS DONE!!!!"
	    echo "============================================="

fi

#get egs
if [ $egs -eq 1 ]; then
	cmd=run.pl
	left_context=16
	right_context=12

	context_opts="--left-context=$left_context --right-context=$right_context"

	# ! [ "$num_hidden_layers" -gt 0 ] && echo \
	#  "$0: Expected num_hidden_layers to be defined" && exit 1;

	  transform_dir=${ali_dir}
	  cmvn_opts="--norm-means=false --norm-vars=false"
	  feat_type=raw
	  extra_opts=()
	  extra_opts+=(--cmvn-opts "$cmvn_opts")
	  extra_opts+=(--online-ivector-dir ${online_ivector_dir})
	  extra_opts+=(--transform-dir $transform_dir)
	  extra_opts+=(--left-context $left_context)
	  extra_opts+=(--right-context $right_context)
	  echo "$0: calling get_egs.sh"


	steps/nnet3/get_egs.sh $egs_opts "${extra_opts[@]}" \
		--num-utts-subset $num_utts_subset \
      --samples-per-iter $samples_per_iter \
      --cmd "$cmd" \
      --frames-per-eg 8 \
      data/${train_set}_hires ${ali_dir} $dir/egs || exit 1;



fi




#train



if [ $train -eq 1 ]; then
cp -r $initialDir/configs $dir
  steps/nnet3/train_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir ${online_ivector_dir} \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs 2 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 5 \
    --trainer.optimization.initial-effective-lrate 0.0017 \
    --trainer.optimization.final-effective-lrate 0.00017 \
    --egs.dir  $dir/egs \
    --cleanup.preserve-model-interval 20 \
    --use-gpu true \
    --feat-dir=data/${train_set}_hires \
    --ali-dir $ali_dir \
    --lang data/lang \
    --reporting.email="$reporting_email" \
    --dir $dir  || exit 1;

fi

graph_dir=exp/tri4/graph_sw1_tg
if [ $decode -eq 1 ]; then


	mfccDev=1
	mfccHiresDev=1
	cvivectors=1
	swbdivectors=0
	#cv_dev_nz cv_test_onlynz test_cslu_cz test_cslu_hi 
	decodeSets="test_vctk_10000"
	for decode_set in $decodeSets ; do


		if [ $mfccDev -eq 1 ]; then
			mfccdir=exp/make_mfcc
		   steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc.conf \
		        --cmd "$train_cmd" data/${decode_set} exp/make_mfcc/${decode_set} $mfccdir;
		    steps/compute_cmvn_stats.sh data/${decode_set} exp/make_mfcc/${decode_set} $mfccdir;

	fi

    if [ $mfccHiresDev -eq 1 ]; then
		    #get the hires mfccs for the dev set
		    		utils/copy_data_dir.sh data/${decode_set} data/${decode_set}_hires
    mfccdir=exp/mfcc_hires
		   steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
		        --cmd "$train_cmd" data/${decode_set}_hires exp/make_hires/${decode_set}_hires $mfccdir;
		    steps/compute_cmvn_stats.sh data/${decode_set}_hires exp/make_hires/${decode_set}_hires $mfccdir;

	fi

    if [ $cvivectors -eq 1 ]; then

      steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    data/${decode_set} exp/nnet3/extractor exp/nnet3/ivectors_${decode_set} || exit 1;

        steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" \
        --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
       $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}_hires || exit 1;

	fi


	if [ $swbdivectors -eq 1 ]; then

	# #get the ivectors of the dev set
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    data/${decode_set}_hires exp/alreadyTrainedModelsOnSwbd/nnet_online/extractor exp/nnet3/ivectors_${decode_set}_hires || exit 1;

    steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" \
        --online-ivector-dir exp/nnet3/ivectors_${decode_set}_hires \
       $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}_hires || exit 1;
   fi
  done

fi


if [ $wer -eq 1 ]; then
for x in ${dir}/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
fi
