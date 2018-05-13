# /exp/data/MozillaCommonVoice/cv_corpus_v1/cv-valid-dev
# /exp/data/MozillaCommonVoice/cv_corpus_v1/cv-valid-test
# /exp/data/MozillaCommonVoice/cv_corpus_v1/cv-valid-train

# inputDir=/exp/data/MozillaCommonVoice/cv_corpus_v1
# outputDir=/exp/abhinav/Mozilla_wav/cv_corpus_v1
# for x in dev test; do
# 	ls $inputDir/cv-valid-$x/* > temp-$x
# 	while read -a line; do
# 		fullfilename=`echo $line | cut -d'/' -f7`
# 		filename=`echo $fullfilename | cut -d'.' -f1`
# 		sox $line -r 16000 $outputDir/cv-valid-$x/${filename}.wav
# 		echo $x-$filename
# 	done < temp-$x

# done
inputFile=$1
mp3Dir=/exp/data/MozillaCommonVoice/cv_corpus_v1
outputDir=/exp/abhinav/Mozilla_wav/cv_corpus_v1/cv-valid-train
while read -a line; do
	temp=`echo $line | cut -d',' -f1`
	fullfilename=`echo $temp | cut -d'/' -f2`
	filename=`echo $fullfilename | cut -d'.' -f1`
	echo $filename
	sox $mp3Dir/$temp -r 16000 $outputDir/${filename}.wav
done < $inputFile
