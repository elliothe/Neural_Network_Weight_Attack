def reorganize_param(model, lr):
	# All parameters except the thresholds
	all_param = [
		param for name, param in model.named_parameters()
		if not 'delta_th' in name
	]

	th_param = [
		param for name, param in model.named_parameters()
		if 'delta_th' in name
	]

	if len(th_param) > 0:
		params = [
			{'params': all_param},
			{'params': th_param, 'weight_decay': 0}
		]
	else:
		params = [
			{'params': all_param}
		]

	return params
