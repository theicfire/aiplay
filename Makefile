clear:
	find . | grep .ipynb | grep -v venv | xargs -I@ nbdev_clean --clear_all --fname @
