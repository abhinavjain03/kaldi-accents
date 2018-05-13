. ./cmd.sh
. ./path.sh

nj=20
modelDirectory=/exp/abhinav/accents_exp
bnfNnetModelDir=/exp/abhinav/accent_recognizer_exp/nnet3/tdnn_relufirst_1024nodes_300bnlayer_nz
# bnfNnetModelDir=/exp/minali/accents_multitask_exp/nnet3/multitask_correct_bignn_0.7_0.3
affix=
train_stage=-10
bnf_dim=1024
singletask=1

# cv_train_all_onlyindian_500_with_accents
x=accent_id_cslu_data
online_ivector_dir=exp/nnet3/ivectors_${x}

. utils/parse_options.sh


mfcc=0
mfcchires=1
ivector=0
bnf=1

# steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
# data/$x  exp/nnet3/extractor ${online_ivector_dir} || exit 1;

bnf_exp_dir=exp
bnf_feat_dir=data/${x}_bnf


if [ $mfcc -eq 1 ]; then

  	mfccdir=exp/mfcc
	steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc.conf --cmd "$train_cmd" \
			data/$x exp/make_mfcc/$x $mfccdir
	steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
	utils/fix_data_dir.sh data/$x


fi

if [ $mfcchires -eq 1 ]; then
	mfccdir=exp/mfcc_hires
	utils/copy_data_dir.sh data/$x data/${x}_hires
	steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf --cmd "$train_cmd" \
			data/${x}_hires exp/make_hires/$x $mfccdir;
	steps/compute_cmvn_stats.sh data/${x}_hires exp/make_hires/${x} $mfccdir;
	utils/fix_data_dir.sh data/${x}_hires
fi

online_ivector_dir=exp/nnet3/ivectors_${x}
if [ $ivector -eq 1 ]; then
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    data/${x} exp/nnet3/extractor $online_ivector_dir || exit 1;
fi

if [ $bnf -eq 1 ]; then

	if [ $singletask -eq 1 ]; then

		steps/nnet3/make_bottleneck_features_text_from_singletask.sh \
		--nj $nj \
		--use-gpu true \
		--cmd "$train_cmd" \
		    tdnn_bn.renorm data/${x}_hires bnf_feat_dir \
		    $bnfNnetModelDir $bnf_exp_dir/make_bnf/${x} $bnf_exp_dir/make_bnf
		else
		 
		  steps/nnet3/make_bottleneck_features_text.sh \
		  --nj $nj \
		  --use-gpu true \
		  --ivector-dir exp/nnet3/ivectors_${x} \
		  --cmd "$train_cmd" \
		      acc_btn.renorm data/${x}_hires $bnf_feat_dir \
		      $bnfNnetModelDir $bnf_exp_dir/make_bnf/${x} $bnf_exp_dir/make_bnf
	fi


fi







# run.pl JOB=1:20 scratch/temp/convert.JOB.log copy-feats ark,t:scratch/make_bnf/test/txt/raw_bnfeat_cv_train_nz_sp_hires.JOB.txt ark,scp:scratch/make_bnf/test/raw_bnfeat_cv_train_nz_sp_hires.JOB.ark,scratch/make_bnf/test/raw_bnfeat_cv_train_nz_sp_hires.JOB.scp
# for n in $(seq $20); do  cat /exp/minali/txt/bnf_feats/raw_bnfeat_cv_train_nz_sp_hires.$n.scp; done > /exp/minali/txt/bnf_feats/feats.scp
