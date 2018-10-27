
submit: submit-LF submit-LFV

submit-LF:
	dts challenges submit --challenge aido1_LF1_r3-v3

submit-LFV:
	dts challenges submit --challenge aido1_LFV_r1-v3



submit-no-cache: 
	$(MAKE) submit-LF-no-cache
	$(MAKE) submit-LFV

submit-LF-no-cache:
	dts challenges submit --challenge aido1_LF1_r3-v3 --no-cache

submit-LFV-no-cache:
	dts challenges submit --challenge aido1_LFV_r1-v3 --no-cache
