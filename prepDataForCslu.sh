ls /exp/data/accentID/cslu_fae/*/* > temp_list



while read -r var
do
  fullName=`basename $var`
  name=`echo $fullName | cut -d'.' -f1`
  echo $name" "$name >> data/accent_id_cslu_data/utt2spk
  echo $name" "$name >> data/accent_id_cslu_data/spk2utt
  echo $name" "$var	 >> data/accent_id_cslu_data/wav.scp
done < temp_list