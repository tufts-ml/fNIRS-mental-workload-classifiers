import numpy as np
import pandas as pd
import scipy.stats
import itertools

from collections import OrderedDict

def iterator__product_of_dict_values(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield OrderedDict(zip(keys, instance))

if __name__ == '__main__':
	my_settings_grid = OrderedDict()
	my_settings_grid['subject_id'] = ['%03d' % a for a in range(5)]
	my_settings_grid['classifier_name'] = [
		'CNN', 'LR']
	my_settings_grid['paradigm_name'] = [
		'subject_specific', 'generic']
	my_settings_grid['n_generic_subjects_used_for_training'] = [0,4,16,64]
	my_settings_grid['frac_of_target_subject_data_used_for_training'] = \
		[0, 1.0]
	row_list = list()

	for S in iterator__product_of_dict_values(**my_settings_grid):

		if S['n_generic_subjects_used_for_training'] == 0:
			if S['frac_of_target_subject_data_used_for_training'] == 0.0:
				continue
		if S['paradigm_name'].count('subject_specific'):
			if S['frac_of_target_subject_data_used_for_training'] == 0.0:
				continue
		if S['paradigm_name'].count('subject_specific'):
			if S['frac_of_target_subject_data_used_for_training'] == 0.0:
				continue
		else:
			if S['frac_of_target_subject_data_used_for_training'] > 0.0:
				continue
		if S['paradigm_name'].count('generic'):
			if S['n_generic_subjects_used_for_training'] == 0:
				continue

		prng = np.random.RandomState(
			(int(S['subject_id'])+1) * 10000 +
			int(S['n_generic_subjects_used_for_training']))

		# Define the mean for this subject
		m = (
			0.50
			+ prng.uniform(low=0.0, high=0.2)
			+ 0.1 * np.sqrt(S['frac_of_target_subject_data_used_for_training'])
			+ 0.05 * np.sqrt(S['n_generic_subjects_used_for_training']) / 8.0
			+ 0.05 * S['classifier_name'].count("CNN")
			+ 0.03 * (
				float(S['paradigm_name'].count("subject_specific"))
				* float(S['n_generic_subjects_used_for_training'] / 64))
		)

		# Define variance, which generally shrinks with more data
		v = (
			0.005
			+ prng.uniform(low=0.0, high=0.001)
			- 0.002 * np.sqrt(S['frac_of_target_subject_data_used_for_training'])
			- 0.001 * np.sqrt(S['n_generic_subjects_used_for_training']) / 8.0
		)
		if v < 0.001:
			v = prng.uniform(low=0.0005, high=0.001)

		# Define Beta with provided mean/variance
		# Source: https://stats.stackexchange.com/questions/12232/calculating-the-parameters-of-a-beta-distribution-using-the-mean-and-variance
		G = (v + m*m - m)
		a = - m * G / v
		b = G * (m-1) / v
		acc_samples = scipy.stats.beta(a,b).rvs(random_state=prng, size=100000)
		dist_below_rand = np.maximum(0.5 - acc_samples, 0.0)
		chance_forget = dist_below_rand**2
		n_to_forget = int(np.ceil(0.95 * np.sum(chance_forget)))
		if n_to_forget == 0:
			valid_acc_samples = acc_samples
		else:
			ids_to_forget = prng.choice(np.arange(acc_samples.size), 
				size=n_to_forget,
				p=chance_forget / np.sum(chance_forget))
			valid_acc_samples = [
				acc_samples[i] for i in range(acc_samples.size)
				if i not in ids_to_forget]

		percentiles_B = np.asarray([10, 25, 50, 75, 90])
		val_at_percentiles_B = np.percentile(valid_acc_samples, percentiles_B)

		for p, acc in zip(percentiles_B, val_at_percentiles_B):
			key = 'acc_at_%dth' % (p)
			S[key] = acc
		row_list.append(S)

	all_df = pd.DataFrame(row_list)
	columns = 'classifier_name,subject_id,paradigm_name,n_generic_subjects_used_for_training,frac_of_target_subject_data_used_for_training,acc_at_10th,acc_at_25th,acc_at_50th,acc_at_75th,acc_at_90th'.split(',')
	all_df = all_df.sort_values(
		columns[:4], ascending=True)

	all_df.to_csv(
		'fakeresults_accuracy_by_subject.csv',
		columns=columns,
		index=False,
		float_format='%.4f')


