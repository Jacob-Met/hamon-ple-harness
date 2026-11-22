.PHONY: check
check:
	cd .. && python3 qualification-fin-161636/course.py . /tmp/fin-161636-$$$$.json && python3 qualification-fin-161636/report.py /tmp/fin-161636-$$$$.json
	cd .. && PYTHONPATH=. python3 -m unittest discover -v -s tests -p test_wave5_pde_control_psychro_fem.py
	cd .. && PYTHONPATH=. python3 -m unittest discover -v -s tests -p test_prescribed_fin_flux.py
