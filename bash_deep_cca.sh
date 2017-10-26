for filename in models/DeepCCA/model_config/*;
do
	echo $filename
	python deep_cca_with_config.py --filename $filename &
done;
