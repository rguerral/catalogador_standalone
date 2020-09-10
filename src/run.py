from catalog import Catalog
import pathlib

if __name__ == "__main__":
	print("-INICIANDO EJECUCIÓN-")
	MAIN_PATH = pathlib.Path(__file__).parent.absolute().parent.absolute()
	a = Catalog(MAIN_PATH = MAIN_PATH)
	a.check_inputs_integrity()
	a.preprocess_inputs()
	a.df_to_json()
	a.predict_categories()
	a.predict_attributes()
	a.export_csv()
	a.export_json()
	print("-EJECUCIÓN COMPLETA-")