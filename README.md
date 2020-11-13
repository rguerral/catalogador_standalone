## 0. Introducción
El siguiente repositorio contiene un algoritmo que estructura la información de productos (catalogador). El catalogador toma la descripción de un producto en formato de texto no estructurado y retorna un diccionario con su categoría, atributos y los valores de estos atributos.
```
Input: "HAMBURGUESA VACUNO CONGELADA TIL SACHET 50 G UNIDAD"
Output:
    * category: "Hamburguesas Y Churrascos"
    * envase: "sachet"
    * marca: "til"
    * contenido (gramo): 50
```

```
Input: "JUGO LIQUIDO  FLORIDAS NATURAL POMELO CAJA 1,75 L UNIDAD"
Output:
    * category: "Jugos"
    * envase: "caja"
    * marca: "floridas natural"
    * sabor: "pomelo"
    * contenido (litro): 1.75
```

## 1. Instalación

Para poder ejecutar el algoritmo es necesario crear un ambiente virtual con ```python 3.6``` y los paquetes incluidos en ```requirements.txt```. Esto se puede hacer de varias formas, de las cuales recomiendo utilizar Anaconda de la siguiente manera:
1. Descargar e instalar Anaconda desde el siguiente [link](https://docs.anaconda.com/anaconda/install/). Seguir los pasos de instalación según sistema operativo. Finalizada la instalación deberías poder ejecutar la app Anaconda Prompt (si usas Windows) o poder ejecutar comandos ```conda``` en Terminal si usas macOS o Linux.
2. Descargar/clonar el repositorio.
2. Abrir Anaconda Prompt (Windows) o Terminal (macOS, Linux).
3. Navegar hasta el directorio del repositorio. Para esto debes ejecutar el comando ```cd "<ruta>"```en donde <ruta> es la ruta hasta el directorio del repositorio, en mi caso es ```cd "/Users/admin/github/catalogador_standalone"```.
4. Crear el ambiente virtual. Para esto ejecutar el comando ```conda create --name catalogador_standalone python=3.6 --file requirements.txt```.
 
## 2. Ejecución
1. Abrir Anaconda Prompt (Windows) o Terminal (macOS, Linux).
2. Navegar hasta el directorio del repositorio. Para esto debes ejecutar el comando ```cd "<ruta>"```en donde <ruta> es la ruta hasta el directorio del repositorio, en mi caso es ```cd "/Users/admin/github/catalogador_standalone"```.
3. Activar el ambiente virtual. Para esto ejecutar el comando ```conda activate catalogador_standalone```.
4. Ejecutar el comando ```python src/run.py```.
5. Ingresar el número del input_state a ejecutar.
5. El algoritmo finaliza su ejecución cuando muestra el mensaje *- EJECUCIÓN COMPLETA -*. El output del algoritmo (productos catalogados) estará en formato .csv y .json en la carpeta ```catalogador_standalone/outputs```

## 3. Datos requeridos (inputs_states/)
>  ***NOTA: Estos archivos deben ser definidos manualmente. Agregué dos ejemplos: para la catalogación de alimentos y computadores. Si se quiere catalogar un nuevo listado de productos se debe crear una carpeta y definir siguientes archivos.***

### 3.1. catalog_data.json
Contiene las posibles categorías y sus atributos. Una categoría puede contener uno o más atributos. Por ejemplo, se puede definir la categoría ```Aceite``` con atributos ```marca```, ```tamaño```. Esta categoría y sus atributos serán utilizados durante toda la sección.

Un atributo a su vez puede ser de dos tipos: nominal o ratio. Los atributos nominales son aquellos que toman valores discretos no ordenables en una escala inherente (p.ej. ```marca```), mientras que los atributos ratio toman valores compuestos por una magnitud númerica y una unidad de medida (p.ej. ```tamaño```, que toma valores como *'1 litro'* o *'0.5 litros'*). 

El ejemplo de la categoría ```Aceite``` quedaría definido así:
```
{
    "Aceite": {
        "marca": {
            "type": "nominal"
        },
        "tamaño":{
            "type": "ratio"
        }
    }
}
```

> **NOTA: Los nombres de los atributos (p.ej. marca o tamaño) serán transformados a minuscula.** 

El diccionario contiene informacion adicional para los atributos según su tipo:
#### 3.1.1. Atributos nominales
Los atributos nominales cuentan con un campo ```possible_values``` que almacena los posibles valores que el atributo puede tomar. Cada posible valor tiene un listado de patrones, que está conformado por sinonimos y/o *expresiones regulares*. El algoritmo buscará si algun patrón del listado de patrones ocurre en el texto y de ser así, asignará el respectivo valor al atributo. Por ejemplo, el atributo ```marca```(de la categoríea ```Aceite```) podría tomar valores como *'campo lindo'* o *'miraflores'*. Esto se agrega al diccionario de la siguiente forma:
```
...
        "marca": {
            "type": "nominal",
            "possible_values": {
                "miraflores": ["miraflores"]
                "campo lindo": ["campo lindo"]
            }
        },
...
```
Supongamos que analizando los productos a catalogar, nos damos cuenta que la marca *'campo lindo'* suele escribirse como *'campolindo'*, *`campo-lindo'* o *'canpo lindo'. Estos valores no serían detectados por el algoritmo al menos que los agreguemos como sinonimos al diccionario de la siguiente forma:
```
...
        "marca": {
            "type": "nominal",
            "possible_values": {
                "miraflores": ["miraflores"]
                "campo lindo": ["campo lindo", "campolindo", "campo-lindo", "canpolindo"]
            }
        },
...
```
También podriamos resumir los sinonimos anteriores en una sola expresión regular:
```
...
        "marca": {
            "type": "nominal",
            "possible_values": {
                "miraflores": ["miraflores"]
                "campo lindo": ["ca(m|n)po(\s|-)?lindo"]
            }
        },
...
```
> **Nota: Los posibles valores de atributos y sus listados de patrones deberán definirse siempre en minuscula (el algoritmo transforma los nombres de productos a minusculas en un comienzo).**

#### 3.1.2. Atributos ratio
Los atributos ratio cuentan con un campo ```dim``` que hace referencia a la dimensión del atributo. Por ejemplo, en la categoría ```Aceite``` el atributo ```tamaño``` corresponde a la dimensión ```volumen```, ya que las unidades en las que se mide son litros, mililitros, etc. (unidades de volumen). El diccionario **units_data.json** contiene información de cada dimensión. Las dimensiones que se asignen al campo ```dim```deben estar definidas en **units_data.json** al momento de ejecutar el algoritmo.

De esta forma, el diccionario completo de la categoría ```Aceite```queda:
```
{
    "Aceite": {
        "marca": {
            "type": "nominal",
            "possible_values": {
                "miraflores": ["miraflores"],
                "campo lindo": ["campo lindo", "campolindo", "campo-lindo", "canpolindo"]
            }
        },
        "tamaño":{
            "type": "ratio",
            "dim": "volumen"
        }
    }
}
```

### 3.2. units_data.json

Contiene las dimensiones permitidas y las posibles unidades de cada una. Por ejemplo, la dimensión ```volumen```tiene unidades como ```litro```, ```mililitro```. También contiene las transformaciones entre unidades (p.ej. 1 mililitro = 0.001 litros). Esto se estructura de la siguiente forma: una dimension siempre tendrá una unidad base, supongamos en este ejemplo que es ```litro```. Todas las unidades posbiles definidas en ```possible_units``` tendrán un campo ```to_base```que corresponde al valor númerico por el cual se debe multiplicar la unidad para transformarla en valor base. **Es necesario que la unidad base ```base_unit``` esté definida en ```possible_units```** ( de lo contrairo ocurrirá un error).

```
{
    "volumen": {
        "base_unit": "litro",
        "possible_units": {
            "litro": {
                "to_base": 1,
                "syn": [
                    "lt", "l"
                ]
            },
            "mililitro": {
                "to_base": 0.001,
                "syn": [
                    "cc"
                ]
            }
        }
    }
}
```
Las unidades también tienen un campo `syn`, que es utilizado al momento de detectar un atributo ratio (análogo al campo ```possible_values``` en los atributos nominales). Si la descripción de un producto contiene *'... 1 litro ....'* o  *'... 500 cc ...'*, el valor será detectado. Si la descripción dice *'... 500 ml ...'* el valor no será detectado, para esto habría que agregar el sinonimo 'ml' al campo `syn`del atributo `mililitro`. 
> **NOTa: Las unidades y sus sinonimos deben ser agregados solo en singular (p.ej. litro y no litros). El algoritmo manejara los plurales de forma automática (i.e. detectará tamién '500 litros').**

> **NOTA: Las unidades y sus sinonimos deben ser agregados siempre en minusculas**

#### 3.3. products.csv
Contiene los productos a catalogar. Cuenta con 3 columnas:
- **id:** Identificador único de un producto. Ejemplos: idProductoCM, *[1,2,3,4,..., n]*.
- **text:** Texto con la descripción del producto que será utilizada para detectar su categoría y atributos. Ejemplos: *"aceite kardamili oliva extra virgen 500 ml"*, *"CEREAL ENLINEA BOLITAS DE CHOLATE CAJA 330 G UNIDAD"*
- **category:** Categoría del producto. Puede ser un valor vacio (```BLANK```, ```NA```) si se quiere que el algoritmo detecte su categoría. **Cada categoría debe tener al menos `param.knn_neigh` (default = 5) productos** o el algoritmo arrojará un error (esto se discute en detalle en **4.**)

|    id   |                      text                     | category |
|:-------:|:---------------------------------------------:|:--------:|
|  968405 | aceite miraflores maravilla 1 l 12   unidades |          |
|  542451 |    ACEITE VEGETAL LOS SILOS BIDON 5 L UNIDA   |  Aceite  |
|  543599 |  JUGO LIQUIDO  AFE MANZANA CAJA 200 CC UNIDAD |   Jugos  |
| 1182023 | mostaza jano bolsa de 1k 10 unidades por caja |          |
|   ...   |                      ...                      |    ...   |

## 4. Funcionamiento y parametros

El objetivo de esta sección es detallar el funcionamiento del algoritmo, y como los distintos parametros definidos en ```src/parameters.py``` influyen en este. Es útil conocer los parametros debido a que variarlos puede entregar un mejor resultado dependiendo de cada caso. Se hará mención a los parametros de la forma ```param.<nombre en parameters.py>```.

El algortimo funciona con dos métodos que ocurren secuencialemente: (i) detectar categorías y (ii) detectar valores de atributos. Para poder ejecutar (ii) es necesario conocer la categoría de un producto (ya sea porque fue ingresada manualmente o fue detectada en (i)).

### 4.1. Detectar categorías
Corresponde a un clasificador kNN (k-nearest neighbors) entrenado sobre los productos del archivo ```products.csv``` que tienen categoría definida. El parametro más relevante de este clasificador es el número de vecinos k que se utilizará (```param.knn_neigh```). Para poder entrenar el clasificador es necesario contar con al menos ```param.knn_neigh``` productos por categoría (definidos en ```products.csv```). Con el clasificador entrenado, se predicen los valores de los productos sin categoría. Las predicciones que entrega el clasificador tienen una métrica de confianza que va desde 0 a 1 (0 poco confiable, 1 muy confiable). Las predicciones con confianza mayor o igual a ```param.threshold_class``` (default = 1) serán aceptadas. Las predicciones que no cumplan con lo anterior serán ignoradas y la categoría del producto permanecerá en blanco (para estos productos no se detectarán sus atributos).

El resto de los parametros ```param.knn_max_features```, ```param.knn_min_df```, ```param.knn_ngram``` y ```param.knn_stopwords``` son parametros del vectorizador TFIDF ([sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)).


> **Recomendaciones**
> 1\. El clasificador funcionará mejor en cuanto mayor sea el set de entrenamiento. Esto se traduce en que entre más productos con ```category``` definida existan en el archivo ```products.csv```, mejor será la detección de categorías para el resto de los productos.
> 2\. El clasificador funcionará mejor si las categorías se encuentran balanceadas. Es decir, si es que en el archivo ```products.csv``` contiene el mismo número de productos por categoría.

### 4.2. Detectar valores de atributos
Lo valores de atributos son detectados según su tipo.
#### 4.2.1 Atributos nominales
Se busca la ocurrencia de los patrones definidos en ```catalog_data.json```. Los patrones deben ocurrir rodeados de los escape_chars definidos en los parametros, de la siguiente forma: ```param.escape_chars_begin + <patron> + param.escape_chars_end```. Los ```param.escape_chars``` corresponden a expresiones regulares. Por default ambos son iguales: 
```param.escape_chars_begin = param.escape_chars_end = "(^|$|\s|\/|,|\.)" ```
Esto fuerza a que el patrón ocurra rodeado de espacios (```\s```) o al inicio de un texto (```^```)o  entre comas (```,```), etc. A modo de ejemplo: el patrón *"miraflores"* en el texto *"arrozmiraflores 1kg"* no será encontrado ya que no existe un ```param.escape_chars_begin```. El patrón *"miraflores"* si se encontrará en los siguientes textos: *"miraflores arroz 1kg"*, *"arroz 1kg miraflores"*, *"arroz miraflores 1kg"*, *"arroz miraflores, 1kg"*.

### 4.2.2. Atributos ratio
Se busca la ocurrencia de las unidades (y sus sinonimos) de las dimension ```dim```del atributo. Se debe cumplir la siguiente expresión regular: 
>```param.escape_chars_begin + param.number_regex + param.escape_chars_between + x + param.optional_plural_regex + param.escape_chars_end```

en donde ```x``` es una unidad o un sinonimo de unidad. Los parametros ```param.escape_chars_begin```y ```param.escape_chars_end```son los mismos definidos en la sección anterior. El resto de los parametros se definen a continuación
* ```param.number_regex```: es la expresion regular para identificar patrones númericos. Detecta numeros enteros (p.ej 5), numeros con decimales (p.ej. 5.5 o 4.242) y divisiones de numeros enteros o decimales (p.ej 1/4 o 8.8/4.2).
    > NOTA: cambiar este parametro muy probablemente requiere cambiar la función ```predict_attribute_single``` de ```catalog.py```
* ```param.escape_chars_between```: seteado por default a ```"(\s)?"``` lo que se traduce a un espacio opcional entre el valor numerico y la unidad. Por ejemplo, se detectara la ocurrencia de *"4 kilo"* y *"4kilo"*, pero no "4-kilos".
* ```param.optional_plural_regex```: seteado en ```"(s|es)?"```, lo que quiere decir que si se define la unidad "kilo" (singular) se buscaran tambien los patrones con unidad "kilos" y "kiloes".

**El algoritmo transformará todos los atributos ratio detectados a la unidad base definida**.

> **Recomendaciones**
> 1\. Definir los patrones siempre en minuscula (atributos nominales).
> 2\. Definir las unidades en minuscula y en singular (atributos ratio).

# 5. Outputs
Una vez ejecutado el algoritmo este entregará dos outputs que contienen practicamente la misma información, solo que en distintos formatos. Estos serán almacenados en la carpeta ```catalogador_standalone/output```.
### 5.1. output.csv
* Tiene las columnas ```id```, ```text```y ```category``` del input ```products.csv```. 
* La columna ```conf```es el valor de la confianza de la predicción entregada por el clasificador. Aquellos productos que se ingresaron con categoría en el input ```products.csv```tendrán ```conf = 99```.
* Existirá una columna por cada atributo definido. Los atributos ratio tendran nombre ```"nombre_definido (unidad_base)"``` y serán valores númericos. Si para un atributo se detecta más de un valor, se incluiran todos los valores detectados separados por ";".

### 5.2. output.json
Contiene la misma información que ```output.csv```, pero adicionalmente cuenta con el campo ```topn_predictions``` que muestra el listado completo de las predicciones de categoría junto a sus niveles de confianza. Por ejemplo:
```
"topn_predictions": [
                        ["Verdura Fresca", 0.4],
                        ["Yogurt",  0.2],
                        ["Dulces",  0.2],
                        ["Jugos",0.2]
                ]
```