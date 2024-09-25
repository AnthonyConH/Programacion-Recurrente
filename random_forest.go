package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"sync"
	"time"
)

// Datos de entrada y etiquetas
type Dataset struct {
	X [][]float64 // Características
	Y []int       // Etiquetas (0 o 1)
}

// Árbol de decisión básico
type DecisionTree struct {
	Threshold float64
	Feature   int
	Left      *DecisionTree
	Right     *DecisionTree
	Label     int
}

// Modelo Random Forest
type RandomForest struct {
	Trees []*DecisionTree
	mu    sync.Mutex // Mutex para evitar condiciones de carrera al acceder a Trees
}

// Función para entrenar el Random Forest
func (rf *RandomForest) Train(data Dataset, numTrees int, maxDepth int) {
	var wg sync.WaitGroup

	for i := 0; i < numTrees; i++ {
		wg.Add(1)

		// Cada árbol se entrena en una goroutine
		go func() {
			defer wg.Done()

			// Crear subconjunto de datos aleatorio (con reemplazo)
			subset := bootstrapSample(data)

			// Crear y entrenar un árbol de decisión
			tree := buildTree(subset, maxDepth)

			// Bloquear el acceso concurrente a rf.Trees
			rf.mu.Lock()
			rf.Trees = append(rf.Trees, tree)
			rf.mu.Unlock()
		}()
	}

	// Esperar a que todos los árboles hayan sido entrenados
	wg.Wait()
}

// Función para hacer predicciones con el Random Forest
func (rf *RandomForest) Predict(sample []float64) int {
	votes := make(map[int]int)
	var mu sync.Mutex
	var wg sync.WaitGroup

	// Paralelizar las predicciones de cada árbol
	for _, tree := range rf.Trees {
		wg.Add(1)

		go func(tree *DecisionTree) {
			defer wg.Done()

			label := tree.Predict(sample)

			// Bloquear el acceso concurrente al mapa de votos
			mu.Lock()
			votes[label]++
			mu.Unlock()
		}(tree)
	}

	// Esperar a que todas las predicciones se completen
	wg.Wait()

	// Votación mayoritaria
	var maxVotes int
	var predictedLabel int
	for label, count := range votes {
		if count > maxVotes {
			maxVotes = count
			predictedLabel = label
		}
	}
	return predictedLabel
}

// Construir un árbol de decisión básico (árbol binario) de manera recursiva
func buildTree(data Dataset, maxDepth int) *DecisionTree {
	// Condición base de recursión
	if maxDepth == 0 || isPure(data) {
		return &DecisionTree{
			Label: mostCommonLabel(data.Y),
		}
	}

	// Encontrar la mejor característica y umbral para dividir los datos
	feature, threshold := bestSplit(data)

	// Dividir los datos en dos subconjuntos
	left, right := splitDataset(data, feature, threshold)

	// Crear un nodo y dividir recursivamente
	return &DecisionTree{
		Threshold: threshold,
		Feature:   feature,
		Left:      buildTree(left, maxDepth-1),
		Right:     buildTree(right, maxDepth-1),
	}
}

// Predicción en un árbol de decisión
func (tree *DecisionTree) Predict(sample []float64) int {
	if tree.Left == nil && tree.Right == nil {
		return tree.Label
	}

	if sample[tree.Feature] < tree.Threshold {
		return tree.Left.Predict(sample)
	} else {
		return tree.Right.Predict(sample)
	}
}

// Función para hacer el muestreo aleatorio (bootstrap) de los datos
func bootstrapSample(data Dataset) Dataset {
	rand.Seed(time.Now().UnixNano())
	var subset Dataset
	for i := 0; i < len(data.X); i++ {
		index := rand.Intn(len(data.X))
		subset.X = append(subset.X, data.X[index])
		subset.Y = append(subset.Y, data.Y[index])
	}
	return subset
}

// Encontrar la mejor característica y umbral para dividir los datos
func bestSplit(data Dataset) (int, float64) {
	rand.Seed(time.Now().UnixNano())
	feature := rand.Intn(len(data.X[0]))
	minVal := data.X[0][feature]
	maxVal := data.X[0][feature]
	for i := range data.X {
		if data.X[i][feature] < minVal {
			minVal = data.X[i][feature]
		}
		if data.X[i][feature] > maxVal {
			maxVal = data.X[i][feature]
		}
	}
	threshold := rand.Float64()*(maxVal-minVal) + minVal
	return feature, threshold
}

// Función para dividir el dataset en dos subconjuntos en función de una característica y un umbral
func splitDataset(data Dataset, feature int, threshold float64) (Dataset, Dataset) {
	var left, right Dataset
	for i := range data.X {
		if data.X[i][feature] < threshold {
			left.X = append(left.X, data.X[i])
			left.Y = append(left.Y, data.Y[i])
		} else {
			right.X = append(right.X, data.X[i])
			right.Y = append(right.Y, data.Y[i])
		}
	}
	return left, right
}

// Verificar si el conjunto de datos tiene una sola etiqueta
func isPure(data Dataset) bool {
	for i := 1; i < len(data.Y); i++ {
		if data.Y[i] != data.Y[0] {
			return false
		}
	}
	return true
}

// Función para obtener la etiqueta más común en el dataset
func mostCommonLabel(labels []int) int {
	labelCount := make(map[int]int)
	for _, label := range labels {
		labelCount[label]++
	}
	var mostCommon int
	var maxCount int
	for label, count := range labelCount {
		if count > maxCount {
			mostCommon = label
			maxCount = count
		}
	}
	return mostCommon
}

// Cargar el dataset desde un archivo CSV y omitir la cabecera
func LoadDataset(filename string) (Dataset, error) {
	file, err := os.Open(filename)
	if err != nil {
		return Dataset{}, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	var data Dataset
	records, err := reader.ReadAll()
	if err != nil {
		return Dataset{}, err
	}

	// Omitir la primera fila (cabecera)
	for _, record := range records[1:] {
		// Leer las primeras 3 columnas como características (float64)
		x := make([]float64, 3)
		for i := 0; i < 3; i++ {
			x[i], err = strconv.ParseFloat(record[i], 64)
			if err != nil {
				return Dataset{}, err
			}
		}

		// Leer la cuarta columna como etiqueta (float64), luego convertir a int (0 o 1)
		yFloat, err := strconv.ParseFloat(record[3], 64)
		if err != nil {
			return Dataset{}, err
		}
		y := int(yFloat)

		data.X = append(data.X, x)
		data.Y = append(data.Y, y)
	}

	return data, nil
}

// Función para calcular la precisión del modelo Random Forest
func (rf *RandomForest) Accuracy(data Dataset) float64 {
	correct := 0
	var mu sync.Mutex
	var wg sync.WaitGroup

	for i, x := range data.X {
		wg.Add(1)
		go func(i int, x []float64) {
			defer wg.Done()

			pred := rf.Predict(x)
			if pred == data.Y[i] {
				mu.Lock()
				correct++
				mu.Unlock()
			}
		}(i, x)
	}

	wg.Wait()
	accuracy := float64(correct) / float64(len(data.Y))
	return accuracy
}

// Función principal para cargar datos y entrenar el modelo Random Forest
func main() {
	start := time.Now()
	// Cargar el dataset desde el archivo dataset.csv
	data, err := LoadDataset("dataset.csv")
	if err != nil {
		log.Fatalf("Error al cargar el dataset: %v", err)
	}

	// Crear y entrenar el modelo Random Forest
	rf := &RandomForest{}
	rf.Train(data, 20, 10) // 10 árboles, profundidad máxima de 5

	// Evaluar la precisión del modelo
	accuracy := rf.Accuracy(data)
	elapsed := time.Since(start)
	fmt.Printf("Precisión del modelo: %.2f%%\n", accuracy*100)
	fmt.Printf("Tiempo de ejecución: %s\n", elapsed)
}
