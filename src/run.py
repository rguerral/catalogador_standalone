from catalog import Catalog
import pathlib
import os
import sys

if __name__ == "__main__":
	MAIN_PATH = pathlib.Path(__file__).parent.absolute().parent.absolute()

	# Input path
	print(" ")
	states_path = os.path.join(MAIN_PATH, "inputs_states")
	states_names = list(os.walk(states_path))[0][1]
	print("Seleccionar input state:")
	for i,state in enumerate(states_names):
		print("   {} : {}".format(i, state))
	state_id = None
	while state_id is None:
		state_id = input("ingresar <numero opcion>+ENTER (para salir CONTROL+C):\n")
		try:
			state_id = int(state_id)
		except:
			state_id = None
		if state_id not in [x[0] for x in enumerate(states_names)]:
			state_id = None
	INPUT_PATH = os.path.join(MAIN_PATH, states_path, states_names[state_id])


	print("-INICIANDO EJECUCIÓN-")
	print(INPUT_PATH)
	a = Catalog(MAIN_PATH = MAIN_PATH, INPUT_PATH = INPUT_PATH)
	a.check_inputs_integrity()
	a.preprocess_inputs()
	a.df_to_json()
	a.predict_categories()
	a.predict_attributes()
	a.export_csv()
	a.export_json()
	print("-EJECUCIÓN COMPLETA-")