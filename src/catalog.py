import json
import pandas as pd
import numpy as np
import traceback
import re
import ast
import sys
from tqdm import tqdm 
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from os.path import join, exists
# Parametros categorizar
from parameters import (threshold_class,
						tfidf_max_features,
						tfidf_min_df,
						tfidf_ngram,
						tfidf_stopwords,
						knn_neigh)
# Parametros detectar atributos
from parameters import (escape_chars_begin,
						escape_chars_between,
						escape_chars_end,
						number_regex,
						optional_plural_regex)

class Catalog:

	def __init__(self, MAIN_PATH, INPUT_PATH):

		self.MAIN_PATH = MAIN_PATH
		self.INPUT_PATH = INPUT_PATH

		(self.catalog_data,
		self.units_data,
		self.products_df) = self.load_inputs()

		self.products_data = None


	# Cargar, comprobar y preprocesar archivos input
	def load_inputs(self):
		"""
		Carga los archivos input desde el MAIN_PATH/input
		"""
		print("* Cargando archivos input")
		INPUT_PATH = self.INPUT_PATH

		# catalog_data
		if not exists(join(INPUT_PATH, "catalog_data.json")):
			raise FileNotFoundError("No se encuentra el archivo catalog_data.json en la carpeta input")
		try:
			with open(join(INPUT_PATH, "catalog_data.json")) as f:
				contents = f.read()
				catalog_data = ast.literal_eval(contents)
				f.close()
		except Exception as e:
			traceback.print_exc()
			print(e)
			raise ValueError("El archivo catalog_data.json tiene un formato incorrecto")

		# units_data
		if not exists(join(INPUT_PATH, "units_data.json")):
			raise FileNotFoundError("No se encuentra el archivo units_data.json en la carpeta input")
		try:
			with open(join(INPUT_PATH, "units_data.json")) as f:
				contents = f.read()
				units_data = ast.literal_eval(contents)
				f.close()
		except Exception:
			traceback.print_exc()
			raise ValueError("El archivo units_data.json tiene un formato incorrecto")

		""" 
		products.csv / products.xlsx
		* busca el csv, si no lo encuentra busca el excel
		* da formato a las columnas: "id"= float, "text" = str, "category"= str
		"""
		if exists(join(INPUT_PATH, "products.csv")):
			products_df = pd.read_csv(join(INPUT_PATH, "products.csv"), encoding = "utf-8")
		elif exists(join(INPUT_PATH, "products.xlsx")):
			products_df = pd.read_excel(join(INPUT_PATH, "products.xlsx"), engine='openpyxl')
		else:
			raise FileNotFoundError("products.csv/products.xlsx: no se encuentra el archivo  en la carpeta input")

		try:
			products_df["id"]
			products_df["text"] = [None if x=="nan" else x for x in products_df["text"].astype(str)]
			products_df = products_df[products_df["text"].notna()].copy()
			if products_df["text"].isna().sum() > 0:
				raise ValueError
			products_df["category"] = [None if x=="nan" else x for x in products_df["category"].astype(str)]
		except:
			msg = "products.csv/products.xlsx: formato incorrecto \nse debe cumplir que: columnas = {id:float, text:str, category:str}"
			raise ValueError(msg)


		return catalog_data, units_data, products_df

	def check_inputs_integrity(self):
		"""
		Revisa que los archivos input catalog_data.json, units_data.json y products.csv estén en un formato correcto
		"""
		print("* Revisando integridad archivos cargados")

		# products_csv
		# Las columnas ["id", "text", "category"] deben existir en products_csv
		condition = all(item in self.products_df.columns for item in ["id", "text", "category"])
		if not condition:
			raise ValueError("   * ERROR: products_csv: las columnas necesarias [id, text, category] no están definidas")

		# Los ids deben ser unicos
		condition = len(self.products_df["id"]) == len(self.products_df["id"].unique())
		if not condition:
			raise ValueError("   * ERROR: products_csv: la columna 'id' no tiene valores unicos")

		# Chequear que las categorias de products_csv existan en catalog_data
		csv_categories = list(self.products_df.category.unique())
		catalog_categories = list(self.catalog_data.keys())
		for category in csv_categories:
			if category not in catalog_categories and not pd.isna(category):
				print("   * WARNING: products.csv, catalog_data.json: categoria '{}' existe en products.csv y no en catalog_data.json. No será utilizada".format(category))
				self.products_df["category"] = [None if x == category else x for x in self.products_df.category] 

		# Cada categoría debe tener al menos knn_neigh producto clasificados
		counter_categories = dict(Counter(self.products_df.category))
		categories_error = []
		for k,v in counter_categories.items():
			if v < knn_neigh:
				categories_error.append([k,v])
		if len(categories_error) > 0:
			print("   * ERROR: products.csv: una o más categoría no cumplen con productos minimos clasificados".format(k,v,knn_neigh))
			for k,v in categories_error:
				print("      - category: '{}', number_products: {}, min_number_products: {}".format(k,v,knn_neigh))
			sys.exit()

		# catalog_data
		# Chequear que las categorias de catalog_data existan en products_csv
		csv_categories = list(self.products_df.category.unique())
		catalog_categories = list(self.catalog_data.keys())
		for category in catalog_categories:
			if category not in csv_categories and not pd.isna(category):
				print("   * WARNING: products.csv, catalog_data.json: categoria '{}' existe en catalog_data.json y no en products.csv. No será utilizada".format(category))

		# Atributos nominales con 'possible_values', atributos ratio con 'dim'
		attributes_error = []
		for category, attributes in self.catalog_data.items():
			for attribute, attribute_spec in attributes.items():
				if attribute_spec["type"] == "nominal":
					if set(["type", "possible_values"]) != set(list(attribute_spec.keys())):
						attributes_error.append([category,attribute])
				elif attribute_spec["type"] == "ratio":
					if set(["type","dim"]) != set(list(attribute_spec.keys())):
						attributes_error.append([category,attribute])
				else:
					attributes_error.append([category,attribute])
		if len(attributes_error) > 0:
			print("   * ERROR: existen atributos con valores incorrecto. Todos los atributos deben tener un 'type' = 'nominal o 'type' = 'ratio'. Atributos 'type' = 'nominal' deben tener una lista de 'possible_values'. Atributos 'type' = 'ratio' deben tener una dimension 'dim'")
			for (category, attribute) in attributes_error:
				print("      - category: '{}', attribute: '{}'".format(category, attribute))
			sys.exit()

		# Atributo ratio tiene unidad no definida en units_data
		ratio_attributes_error = []
		dimentions = self.units_data.keys()
		for category, attributes in self.catalog_data.items():
			for attribute, attribute_spec in attributes.items():
				if attribute_spec["type"] == "ratio" and attribute_spec["dim"] not in dimentions:
					ratio_attributes_error.append([category, attribute, attribute_spec["dim"] ])
		if len(ratio_attributes_error) > 0:
			print("   * ERROR: existen atributos ratio con dimensiones no definidas en units_data.json")
			for (category, attribute, dim) in ratio_attributes_error:
				print("      - category: '{}', attribute: '{}', dim: '{}'".format(category, attribute, dim))
			sys.exit()

		# Atributos nominales tienen un listado de valores en mal formato
		##  (attribute_spec["possible_values"] existe. chequeado antes)
		nominal_attributes_error = []
		for category, attributes in self.catalog_data.items():
			for attribute, attribute_spec in attributes.items():
				if attribute_spec["type"] == "nominal":
					pv = attribute_spec["possible_values"]
					if type(pv)==dict:
						for x in [k for k,v in pv.items() if type(v) != list]:
							nominal_attributes_error.append([category, attribute, pv])
					else:
						nominal_attributes_error.append([category, attribute, pv])
		if len(nominal_attributes_error) > 0:
			print("   * ERROR: existen atributos nominales con 'possible_values' en formato incorrecto")
			for (category, attribute, pv) in nominal_attributes_error:
				print("      - category: '{}', attribute: '{}', possible_values: '{}'".format(category, attribute, pv))
			sys.exit()


		# units_data
		# Las dimensiones tienen un formato correcto
		dim_error = []
		for dim, units in self.units_data.items():
			if set(units.keys()) != set(["base_unit", "possible_units"]):
				dim_error.append([dim])
		if len(dim_error) > 0:
			print("   * ERROR: units_data.csv: Las dimensiones deben ser diccionarios con campos 'base_unit' y 'possible_units'")
			for (dim, unit) in possible_units_error:
				print("      - dim: '{}', unit: '{}'".format(dim, unit))
			sys.exit()

		# 'possible_units' tienen un formato incorrecto
		possible_units_error = []
		for dim, units in self.units_data.items():
			for unit, unit_spec in units["possible_units"].items():
				if (set(unit_spec.keys()) == set(["to_base", "syn"]) and 
					type(unit_spec["to_base"]) in [float, int] and 
					type(unit_spec["syn"]) == list and
					len([x for x in unit_spec["syn"] if type(x)!=str]) == 0):
					pass # formato correcto
				else:
					possible_units_error.append([dim, unit])
		if len(possible_units_error) > 0:
			print("   * ERROR: units_data.csv: El campo 'possible_units' de alguna dimension está mal definido. Revisar que el campo 'to_base' sea numerico y el campo 'syn' una lista de strings")
			for (dim, unit) in possible_units_error:
				print("      - dim: '{}', unit: '{}'".format(dim, unit))
			sys.exit()

		# 'base_unit' no existe en 'possible_units'
		base_units_error = []
		for dim, units in self.units_data.items():
			if units["base_unit"] not in units["possible_units"].keys():
				base_units_error.append(dim, units["base_unit"])
		if len(base_units_error) > 0:
			print("   * ERROR: units_data.csv: La unidad base 'base_unit' de alguna dimension no está definida en su campo 'possible_units")
			for (dim, unit) in base_units_error:
				print("      - dim: '{}', base_unit: '{}'".format(dim, unit))
			sys.exit()

	def preprocess_inputs(self):
		# preprocess regex text self.units_data
		for dim, dim_spec in self.units_data.items():
			for unit, unit_spec in dim_spec["possible_units"].items():
				self.units_data[dim]["possible_units"][unit]["syn"] = [self.preprocess_text_regex(x) for x in unit_spec["syn"]]

		# preprocess regex text self.catalog_data
		for category, attributes in self.catalog_data.items():
			for attribute, attribute_spec in attributes.items():
				if attribute_spec["type"] == "nominal":
					for value, regex_list in attribute_spec["possible_values"].items():
						self.catalog_data[category][attribute]["possible_values"][value] = [self.preprocess_text_regex(x) for x in regex_list]

		# pasar nombres de atributos a minuscula
		all_categories_attributes = []
		for category,attributes in self.catalog_data.items():
			for attribute, attribute_spec in attributes.items():
				verbose_name = " ".join(str(attribute).lower().split())
				all_categories_attributes.append([category, attribute, verbose_name])
		for category, attribute, verbose_name in all_categories_attributes:
			self.catalog_data[category][verbose_name] = self.catalog_data[category].pop(attribute)

		# cambiar nombre de atributos ratio para que incluyan la unidad base
		attributes_to_rename = []
		for category,attributes in self.catalog_data.items():
			for attribute, attribute_spec in attributes.items():
				if attribute_spec["type"] == "ratio":
					dim = attribute_spec["dim"]
					verbose_name = "{} ({})".format(attribute, self.units_data[dim]["base_unit"])
					attributes_to_rename.append([category, attribute, verbose_name])
		for category, attribute, verbose_name in attributes_to_rename:
			self.catalog_data[category][verbose_name] = self.catalog_data[category].pop(attribute)


	def df_to_json(self):
		"""
		Transforma products_df a un diccionario json con la estructura:

		{
		# Ejemplo producto con categoria
		id1: {
			text: "CAFÉ INSTANTANEO NESCAFÉ",
			category: "Té, Café",
			attributes: {marca: None,
						 contenido: None,
						...}
			},
		
		# Ejemplo productos sin categoria
		id2: {
			text: "aceite miraflores maravilla 1 l unidad",
			category: None,
			attributes: {}
		}
		"""
		print("* Transformando csv a diccionario")
		products_data = {}
		for idx, row in self.products_df.iterrows():
			id_prod = row["id"]
			text = row["text"]
			category = row["category"]
			if pd.isna(category):
				category = None
				attributes = {}
			else:
				try:
					attributes = {k: None for k in self.catalog_data[category].keys()}
				except:
					attributes = {}
			products_data[id_prod] = {
				"text": text,
				"category": category,
				"attributes": attributes
				}
		self.products_data = products_data

	# Predecir categoria
	def classify_knn(self, data_train, data_test):
		"""
		Clasificador k-Nearest Neighbors

		INPUTS:
		* data_train: subconjunto de productos con categoría en products_data
		* data_train: subconjunto de productos sin categoría en products_data

		OUTPUTS:
		* id_test: ids de productos en el test set 
			ej. [28423, 12324, ...]
		* Y_test_pred: categorías predichas para los productos del test set (la que tiene mayor nivel de confianza)
			ej. ["Aceite", "Té, Café", ... ]
		* Y_test_pred_prob: nivel de confianza de la prediccion de categoría (la que tiene mayor nivel de confiaza)
			ej. [0.8, 1, ...]
		* top_categories_confidences: nivel de confianza para todas las predicciones de categorias (no solo la con mayor nivel de confianza)
			ej. [[(Aceite, 0.8),(Vinagre, 0.6)], [(Té, Café, 1)]]

		"""
		print("* Ejecutando kNN")
		# Preprocess
		id_train = []
		id_test = []
		X_train = []
		X_test = []
		Y_train = []
		Y_test = []
		for k,v in data_train.items():
			X_train.append(self.preprocess_text_classify(v["text"]))
			Y_train.append(v["category"])
			id_train.append(k)
		for k,v in data_test.items():
			X_test.append(self.preprocess_text_classify(v["text"]))
			Y_test.append(v["category"])
			id_test.append(k)
		
		# TF-IDF
		tfidf = TfidfVectorizer(sublinear_tf = True,
								max_features = tfidf_max_features,
								min_df = tfidf_min_df,
								norm = "l2",
								ngram_range = (1,tfidf_ngram),
								stop_words = tfidf_stopwords
								)
		tfidf.fit_transform(X_train)
		X_train = tfidf.transform(X_train)
		X_test = tfidf.transform(X_test)
		
		# kNN
		# Fit
		neigh = KNeighborsClassifier(n_neighbors=knn_neigh)
		neigh.fit(X_train, Y_train)
		# Get list of n closest neigh classes
		neigh_idx = neigh.kneighbors(X_test,  n_neighbors=knn_neigh)[1]
		flat_neigh_idx = np.reshape(neigh_idx, neigh_idx.shape[0]*neigh_idx.shape[1])
		flat_neigh_categories = np.take(np.array(Y_train),flat_neigh_idx)
		neigh_categories = np.reshape(flat_neigh_categories, neigh_idx.shape)
		# Get top n classes & confidence
		top_categories_confidences = []
		for x in neigh_categories:
			tuples = list(Counter(x).items())
			tuples = [(x,round(y/knn_neigh,4)) for x, y in tuples]
			tuples = sorted(tuples, key=lambda tup: tup[1], reverse = True)
			top_categories_confidences.append(tuples)
		# Only top category
		Y_test_pred = list(neigh.predict(X_test))
		Y_test_pred_prob = [round(max(x),4) for x in neigh.predict_proba(X_test)]

		return id_test, Y_test_pred, Y_test_pred_prob, top_categories_confidences

	def predict_categories(self):
		"""
		Detecta la categoría de los productos sin categoría utilizando el algoritmo kNN con los productos con categoría como set de entrenamiento
		OUTPUT:
			update self.products_data
		"""

		# Split productos con categoria y sin categoria
		products_train = {k:v for k,v in self.products_data.items() if v["category"]!=None}
		products_test = {k:v for k,v in self.products_data.items() if v["category"]==None}

		if len(products_test) == 0:
			print("* Saltando predecir categorias: todos los productos están categorizados")
			return
		else:
			print("* Prediciendo categorias")

		# Run kNN
		id_test, Y_test_pred, Y_test_pred_prob, top_categories_confidences = self.classify_knn(products_train, products_test)

		# Seleccionar clasificados automaticamente (confidence over threshold)
		idx_over_threshold = [x >= threshold_class for x in Y_test_pred_prob]
		classified_id = np.array(id_test)[np.array(idx_over_threshold)]
		classified_pred = np.array(Y_test_pred)[np.array(idx_over_threshold)]
		classified_conf = np.array(Y_test_pred_prob)[np.array(idx_over_threshold)]
		classified_topn = np.array(top_categories_confidences, dtype=object)[np.array(idx_over_threshold)]
		# Actualizar clasificados automaticamente. Fijar categorias y atributos
		for id_cm, category, conf in zip(classified_id, classified_pred, classified_conf):
			# Actualizar categorias
			self.products_data[id_cm]["category"] = category
			attributes = {k: None for k in self.catalog_data[category].keys()}
			self.products_data[id_cm]["attributes"] = attributes
		# Agregar topn predicciones
		for id_cm, topn in zip(id_test, top_categories_confidences):
			self.products_data[id_cm]["topn_predictions"] = topn


	# Predecir atributos
	def predict_attribute_single(self, text, category, attribute):
		"""
		Entrega los valores del atributo 'attribute' detectados en el texto 'text'
		
		ej atributo nominal:
		INPUTS:
			text: "Notebook toshiba lenovo 256 gb"
			attribute: "marca"
		OUTPUTS:
			detected_values: ["Toshiba", "Lenovo"]

		ej atributo ratio:
		INPUTS:
			text: "Notebook toshiba lenovo 256 gb 1 tb"
			attribute: "disco duro"
			(base unit = "gb")
		OUTPUTS:
			detected_values: [256, 1000]
		"""

		pptext = self.preprocess_text_attributes(text)
		attribute_data = self.catalog_data[category][attribute]
		
		# Nominal attributes
		if attribute_data["type"] == "nominal":
			detected_values = []
			for value,regex_list in attribute_data["possible_values"].items():
				for x in regex_list:
					regex = re.compile(escape_chars_begin + x + escape_chars_end)
					if len(re.findall(regex, pptext)) > 0 and value not in detected_values:
						detected_values.append(value)
			return detected_values

		# Ratio attributes
		elif attribute_data["type"] == "ratio":
			detected_values = []
			dim = attribute_data["dim"]
			possible_units = self.units_data[dim]["possible_units"]
			base_unit = self.units_data[dim]["base_unit"]

			# Buscar patrones float + unidad.
			# p.ej. "Notebook 8 GB"
			# Se busca la unidad en singulares ("GB") y plural ("GBS")
			for unit, unit_spec in possible_units.items():
				regex_list = unit_spec["syn"]
				mult = unit_spec["to_base"]
				magnitudes = []
				for x in regex_list:
					regex = re.compile(escape_chars_begin + number_regex + escape_chars_between + x + optional_plural_regex + escape_chars_end)
					found_magnitudes = [y[1] for y in re.findall(regex, pptext)]
					# Pasar magnitudes a float
					found_magnitudes = [re.sub(",",".",magnitude) for magnitude in found_magnitudes]
					float_found_magnitudes = []
					for magnitude in found_magnitudes:
						if "/" not in magnitude:
							float_found_magnitudes.append(float(magnitude))
						else:
							float1 = str(float(magnitude.split("/")[0]))
							float2 = str(float(magnitude.split("/")[1]))
							if float(float2)==0:
								float_found_magnitudes.append(float(float1))
							else:
								float_found_magnitudes.append(float(float1) / float(float2))
					# Guardar
					for magnitude in float_found_magnitudes:
						magnitude_base = magnitude * mult
						if magnitude_base not in magnitudes:
							magnitudes.append(magnitude_base)
				detected_values += magnitudes
				

			# Buscar patrones unidad(singular).
			# p.ej. "POSTA PALETA KG." o "Platano granel unidad"
			# Se busca la unidad solo en singular ("kg", "unidad"), sin ser antecedida por un numero
			if len(detected_values) == 0:
				for k,v in possible_units.items():
					unit = k
					regex_list = v["syn"]
					mult = v["to_base"]
					magnitudes = []
					for x in regex_list:
						regex = re.compile(escape_chars_begin + unit + escape_chars_end)
						if len(re.findall(regex, pptext)) > 0:
							if 1*mult not in magnitudes:
								magnitudes.append(1*mult)
				detected_values += magnitudes
					
			return detected_values

	def predict_attributes(self):
		"""
		Detecta y guarda los valores de atributos para los productos (products_data) con categoría
		"""
		print("* Detectando valores de atributos")
		for id_product, product_spec in tqdm(self.products_data.items()):
			if product_spec["category"] == None:
				continue
			attributes = list(product_spec["attributes"].keys())
			for attribute in attributes:
				possible_values = self.predict_attribute_single(
					product_spec["text"],
					product_spec["category"],
					attribute)
				self.products_data[id_product]["attributes"][attribute] = possible_values


	# Preprocesar texto
	def preprocess_text_regex(self, text):
		"""
		Funcion para preprocesar las expresiones regulares ingresadas en catalog_data.json
		"""
		text = str(text)
		text = text.lower()
		text = unidecode(text)
		return text

	def preprocess_text_classify(self, text):
		"""
		Funcion para preprocesar el texto antes de clasificar
		"""
		text = text.lower()
		text = re.sub("\:", " ", text)
		text = re.sub("\.","",text)
		text = re.sub(",","",text)
		text = unidecode(text)
		text = " ".join([x for x in text.split() if not x.isnumeric()])
		return text

	def preprocess_text_attributes(self, text):
		"""
		Funcion para preprocesar el texto antes de detectar atributos
		"""
		text = text.lower()
		text = unidecode(text)
		text = re.sub("\:", " ", text)
		text = " ".join(text.split())
		return text


	# Export
	def export_csv(self, path = None):
		"""
		Exporta el diccionario self.products_data a un archivo .csv.
		El archivo .csv posee menos información que el diccionario completo. En particular, no tiene el campo 'topn_conf'
		"""
		print("* Exportando csv")
		if path == None:
			path = join(self.MAIN_PATH, "output")
		records = []
		for k,v in self.products_data.items():
			idprod = k
			text = v["text"]
			category = v["category"]
			try:
				top_conf = max([x[1] for x in v["topn_predictions"]])
			except:
				top_conf = 99
			record = {"id": idprod,
				"text": text,
				"category": category,
				"conf": top_conf}
			for attribute in list(v["attributes"].keys()):
				record[attribute] = "; ".join([str(x) for x in v["attributes"][attribute]])
			records.append(record)
		df = pd.DataFrame.from_records(records)
		df.to_csv(join(path,"output.csv"),index = False)

	def export_json(self, path = None):
		"""
		Exporta el diccionario self.products_data a un archivo .json
		"""
		print("* Exportando json")
		if path == None:
			path = join(self.MAIN_PATH, "output")
		with open(join(path,"output.json"), 'w') as fp:
			json.dump(self.products_data, fp,  indent=8, sort_keys=True, ensure_ascii=False)


"""
if __name__ == "__main__":
	a = Catalog(MAIN_PATH = "/Users/RodrigoGuerra/git/catalogador_standalone")
	a.check_inputs_integrity()
	a.preprocess_inputs()
	a.df_to_json()
	a.predict_categories()
	a.predict_attributes()
	a.export_csv()
	a.export_json()
"""

