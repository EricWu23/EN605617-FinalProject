# Add your Source files to this variable
SOURCESLOCATION =linear.cu \
      main_mnist.cu \
      mse.cu \
      relu.cu \
      sequential.cu \
      train.cu \
      validate.cu \
      $(UTILS)/utils.cpp \
      $(DATA)/read_csv.cpp
# Add your Source files to this variable
SOURCES = linear.cu \
      main_mnist.cu \
      mse.cu \
      relu.cu \
      sequential.cu \
      train.cu \
      validate.cu \
      utils.cpp \
      read_csv.cpp
# Add your files to be cleaned but not part of the project
IRRELEVANT =

# Add your include paths to this variable
UTILS:=../utils
DATA:=../data

INCLUDES = $(UTILS) \
           $(DATA)
