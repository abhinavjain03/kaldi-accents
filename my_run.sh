. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

nj=20

mfcc=1
alignSpecific=1
tri1=0
tri1decode=0
tri2=0
tri2decode=0
tri3=0
tri3decode=0
tri4=0
tri4decode=0
mfcchirestrain=0
alignalldata=0
tdnn=0
multitask=0
wer=0


mfccdir=exp/mfcc
if [ $mfcc -eq 1 ]; then
	for x in cv_train_onlyindian; do
		steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_16k.conf --cmd "$train_cmd" \
				data/$x exp/make_mfcc/$x $mfccdir
		steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
		utils/fix_data_dir.sh data/$x
	done
fi

if [ $alignSpecific -eq 1 ]; then
  data=cv_train_onlyindian
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
      data/$data data/lang exp/tri4 exp/tri4_${data}_ali || exit 1

fi


if [ $tri1 -eq 1 ]; then
	steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
                      data/cv_train_nz data/lang_nosp exp/mono

    steps/align_si.sh --nj $nj --cmd "$train_cmd" \
                    data/cv_train_nz data/lang_nosp exp/mono exp/mono_ali

  	steps/train_deltas.sh --cmd "$train_cmd" \
                        3200 30000 data/cv_train_nz data/lang_nosp exp/mono_ali exp/tri1
fi

if [ $tri1decode -eq 1 ]; then
	
    graph_dir=exp/tri1/graph_nosp_sw1_tg
    $train_cmd $graph_dir/mkgraph.log \
               utils/mkgraph.sh data/lang_nosp_sw1_tg exp/tri1 $graph_dir
    steps/decode_si.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.config \
                       $graph_dir data/cv_dev_nz exp/tri1/decode_cv_dev_nz_nosp_sw1_tg

fi

if [ $tri2 -eq 1 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
                    data/cv_train_nz data/lang_nosp exp/tri1 exp/tri1_ali
  steps/train_deltas.sh --cmd "$train_cmd" \
                        4000 70000 data/cv_train_nz data/lang_nosp exp/tri1_ali exp/tri2
fi

if [ $tri2decode -eq 1 ]; then

    graph_dir=exp/tri2/graph_nosp_sw1_tg
    $train_cmd $graph_dir/mkgraph.log \
               utils/mkgraph.sh data/lang_nosp_sw1_tg exp/tri2 $graph_dir
    steps/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.config \
               $graph_dir data/cv_dev_nz exp/tri2/decode_dev_nz_nosp_sw1_tg
  	

fi

if [ $tri3 -eq 1 ]; then

	steps/align_si.sh --nj $nj --cmd "$train_cmd" \
        data/cv_train_nz data/lang_nosp exp/tri2 exp/tri2_ali

	steps/train_lda_mllt.sh --cmd "$train_cmd" \
		6000 140000 data/cv_train_nz data/lang_nosp exp/tri2_ali exp/tri3

fi

if [ $tri3decode -eq 1 ]; then

	
	graph_dir=exp/tri3/graph_nosp_sw1_tg
	$train_cmd $graph_dir/mkgraph.log \
		utils/mkgraph.sh data/lang_nosp_sw1_tg exp/tri3 $graph_dir
	steps/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.config \
		$graph_dir data/cv_dev_nz exp/tri3/decode_cv_dev_nz_nosp_sw1_tg
	

fi


if [ $tri4 -eq 1 ]; then
  # Train tri4, which is LDA+MLLT+SAT, on all the (nodup) data.
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
                       data/cv_train_nz data/lang exp/tri3 exp/tri3_ali

  steps/train_sat.sh  --cmd "$train_cmd" \
                      11500 200000 data/cv_train_nz data/lang exp/tri3_ali exp/tri4

fi

if [ $tri4decode -eq 1 ]; then

    graph_dir=exp/tri4/graph_sw1_tg
    $train_cmd $graph_dir/mkgraph.log \
               utils/mkgraph.sh data/lang_sw1_tg exp/tri4 $graph_dir
    steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" \
                          --config conf/decode.config \
                          $graph_dir data/cv_dev_nz exp/tri4/decode_cv_dev_nz_sw1_tg


fi


if [ $mfcchirestrain -eq 1 ]; then
	mfccdir=exp/mfcc_hires
	for x in cv_train_onlyindian; do
		utils/copy_data_dir.sh data/$x data/${x}_hires
		steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires_16k.conf --cmd "$train_cmd" \
				data/${x}_hires exp/make_hires/$x $mfccdir;
		steps/compute_cmvn_stats.sh data/${x}_hires exp/make_hires/${x} $mfccdir;
		utils/fix_data_dir.sh data/${x}_hires
	done
fi



if [ $alignalldata -eq 1 ]; then



steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
      data/cv_train_nz data/lang exp/tri4 exp/tri4_cv_train_nz_ali || exit 1

fi


if [ $tdnn -eq 1 ]; then


  basic=1
  multitask=0
  scaling=0
  

  if [ $basic -eq 1 ]; then
    local/nnet3/run_tdnn.sh --affix redone_baseline --stage 11 --train-stage -10 
  fi


  if [ $multitask -eq 1 ]; then
    
    num_accents=16
    accents_file=cv-train-accents

    affix=multitask

    local/nnet3/run_tdnn_added_multitask.sh --stage 9 --train-stage -10 \
    --num_accents ${num_accents} --accents-file ${accents_file} \
    --affix $affix

  fi

  if [ $scaling -eq 1 ]; then

      for i in 1.5 2.0 3.0; do

        scale_factor=$i
        accent=australia
        affix=${accent}_scaled_all_${scale_factor}

      	local/nnet3/run_tdnn_added_scaling.sh --stage 9 --train-stage -10 \
        --scale true --scale-factor $scale_factor --scale-file ${accent}_accents_train \
        --affix $affix
      done

    fi

fi


if [ $multitask -eq 1 ]; then

	local/nnet3/run_tdnn_multilingual.sh

fi


if [ $wer -eq 1 ]; then
  #for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 
  for x in exp/*/*redone_baseline*/*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
  #for x in exp/*/*/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
fi

