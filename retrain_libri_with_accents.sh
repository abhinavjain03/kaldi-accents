. ./cmd.sh
. ./path.sh

# ./test --num-pdfs=9054 --num-accents=16 --frame-subsampling-factor=1 --online-ivectors=scp:exp/nnet3/ivectors_cv_train_nz_sp/ivector_online.scp --online-ivector-period=10 --left-context=16 --right-context=12 --compress=true --num-frames=8 "ark,s,cs:utils/filter_scp.pl --exclude exp/nnet3/tdnn_d_sp/egs/valid_uttlist data/cv_train_nz_sp_hires/split6/1/feats.scp | apply-cmvn --norm-means=false --norm-vars=false --utt2spk=ark:data/cv_train_nz_sp_hires/split6/1/utt2spk scp:data/cv_train_nz_sp_hires/split6/1/cmvn.scp scp:- ark:- |" "ark,s,cs:filter_scp.pl data/cv_train_nz_sp_hires/split6/1/utt2spk exp/nnet3/tdnn_d_sp/egs/ali.scp | ali-to-pdf exp/tri4_cv_train_nz_ali_sp/final.mdl scp:- ark:- | ali-to-post ark:- ark:- |" scp:cv-train-accents ark:test123.ark



#this directory contains nnet3 trained on all the the complete swbd data
dir=exp/nnet3/tdnn_libri_100_finetuning_cv_train_sp
#dir=exp/alreadyTrainedModelsOnSwbd/tdnn_d_sp


#for egs
num_utts_subset=512 #default is 512
samples_per_iter=400000 #default is 400000



mkdir $dir
#initialDir=exp/nnet3/tdnn_d_sp
libriModelDir=/home/minali/exp/kaldi/librispeech/s5/exp/nnet3_cleaned
initialDir=$libriModelDir/tdnn_sp_100
model=$initialDir/final.mdl
langdir=/home/minali/exp/kaldi/librispeech/s5/data/lang
tri4dir=/home/minali/exp/kaldi/librispeech/s5/exp/tri4b



train_stage=3 #3
#train_set=test_vctk_I_NI_SA_AU_tmp
train_set=cv_train_nz
nj=30

merge=0
initialize=0
calculateMfcc=0
alignments=0
calculateHires=0
ivector=0
egs=0
train=0
decode=1
wer=1
#AR,IN,FR,KO,MA,TA
combine_set="$data/train_cslu_AR $data/train_cslu_FR $data/train_cslu_IN $data/train_cslu_MA $data/train_cslu_KO $data/train_cslu_TA"
if [ $merge -eq 1 ]; then

utils/combine_data.sh $data/${train_set} $combine_set


# utils/subset_data_dir.sh --first data/train_vctk 30000 data/train_vctk_30000
# utils/subset_data_dir.sh --last data/train_vctk 10000 data/test_vctk_10000
# echo "1"

utils/data/modify_speaker_info.sh --utts-per-spk-max 2 $data/${train_set} $data/${train_set}_max2

echo "2"

utils/fix_data_dir.sh $data/${train_set}_max2

echo "3"
rm -r $data/${train_set}
mv $data/${train_set}_max2 $data/${train_set}

echo "4"

echo "============================================="
echo "MERGING OF DATA DONE!!!!"
echo "============================================="
fi


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
ali_dir=exp/tri4_libri_${train_set}_ali
if [ $alignments -eq 1 ]; then

	steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/${train_set} $langdir $tri4dir ${ali_dir}
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
online_ivector_dir=exp/nnet3/ivectors_libri_${train_set}_hires
if [ $ivector -eq 1 ]; then

	  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
	    data/${train_set}_hires  $libriModelDir/extractor ${online_ivector_dir} || exit 1;
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
    --trainer.optimization.num-jobs-final 4 \
    --trainer.optimization.initial-effective-lrate 0.0017 \
    --trainer.optimization.final-effective-lrate 0.00017 \
    --egs.dir  $dir/egs \
    --cleanup.preserve-model-interval 20 \
    --use-gpu true \
    --feat-dir=data/${train_set}_hires \
    --ali-dir $ali_dir \
    --lang $langdir \
    --reporting.email="$reporting_email" \
    --dir $dir  || exit 1;

fi

graph_dir=$tri4dir/graph_nosp_tgsmall
if [ $decode -eq 1 ]; then

	libriivectors=1
	cvivectors=0
	swbdivectors=0


	for decode_set in cv_forlibri_dev_nz cv_forlibri_test_onlynz; do
		utils/copy_data_dir.sh data/${decode_set} data/${decode_set}_hires
    mfccdir=exp/mfcc_hires

    #get the hires mfccs for the dev set
   steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires_16k.conf \
        --cmd "$train_cmd" data/${decode_set}_hires exp/make_hires/${decode_set}_hires $mfccdir;
    steps/compute_cmvn_stats.sh data/${decode_set}_hires exp/make_hires/${decode_set}_hires $mfccdir;


    if [ $libriivectors -eq 1 ]; then

      steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    data/${decode_set}_hires $libriModelDir/extractor exp/nnet3/ivectors_libri_${decode_set} || exit 1;

        steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" \
        --online-ivector-dir exp/nnet3/ivectors_libri_${decode_set} \
       $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}_hires || exit 1;

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
for x in ${exp}/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
for x in ${exp}/*/*/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
fi
