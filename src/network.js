'use strict'

let jStat = require('jstat').jStat
let Matrix = require('./matrix')
let sprintf = require('sprintf').sprintf
let random = require('./random')
let math = require('./math')
let mathjs = require('mathjs')
let mnist = require('mnist')
let _ = require('lodash')

function printMatrix(m, label) {
    if (m instanceof jStat) {
        m = Array.prototype.slice.call(m)
    }

    if (m instanceof mathjs.type.Matrix) {
        m = m._data
    }

    label = label || ""
    let pad = label.split("").map(s => ' ').join("")
    return m.map((row, i) => {
        if (!Array.isArray(row)) {
            return `${(i ? pad : label)} ${row} <-- warning: this is not a matrix`
        }

        return (i ? pad : label) + '[ ' + row.map(e => sprintf('%0.03f', e)).join(', ') + ' ]'
    }).join('\n')
}

function isMatrix(m) {
    // Returns true if m is an array and it's first element is also an
    // array. It would be more thorough to do something like m.every(e =>
    // Array.isArray(e)) but this will probably be faster and just as good
    // in practice.
    return Array.isArray(m) && Array.isArray(m[0])
}

function arrayToMatrix(arr, type) {
    // Converts a simple array to a matrix. The "type" argument describes
    // whether the result should look like a column vector or a row vector.
    type = type || 'column'
    if (type === 'column') {
        return Matrix.create(arr.length, 1, (i, j) => arr[i])
    } else if (type === 'row') {
        return Matrix.create(1, arr.length, (i, j) => arr[j])
    } else {
        throw new Error(`Unknown matrix type ${type}`)
    }
}

class Network {

    /**
     * @param sizes - Array of layer sizes
     */
    constructor(sizes) {
        this.sizes = sizes

        // biases is an array of bias vectors. The input layer obviously
        // does not have a bias vector which is why we are slicing sizes
        // from index 1. biases[i] = [ [b1], [b2], ..., [bN] ], which is the
        // bias vector for layer i. The bias vector is a column vector which
        // is why it is an array of single-element arrays.
        //
        // jStat.rand(rows, cols) creates a matrix of the given row and
        // column dimensions where each element is pulled from the Normal
        // distribution.
        this.biases = this.sizes.slice(1).map(layerSize => Matrix.rand(layerSize, 1))

        // weights is an array of weight matrices. There is one weight
        // matrix "between" each layer. For a given weight matrix
        // weights[i], the column indices map to the different neurons in
        // the incoming "X" layer; the row indices map to the different
        // neurons in the output "Y" layer. Example:
        //
        // weights[i] = [
        //  [ w_x_0y_0, w_x_1y_0, w_x2_y0 ],
        //  [ w_x_0y_1, w_x_1y_1, w_x2_y1 ],
        //  [ w_x_0y_2, w_x_1y_2, w_x2_y2 ]
        // ]
        //
        // w_x2_y1 should be interpreted as, "the weight of the connection
        // between neuron x2 in the X input layer and neuron y1 in the Y
        // output layer."
        //
        // See comment above regarding jStat.rand()
        this.weights = this.sizes.slice(0, -1).map((xLayerSize, i) => {
            let yLayerSize = this.sizes[i + 1]
            return Matrix.rand(yLayerSize, xLayerSize)
        })

        // this.biases.forEach(biasMatrix => console.log(biasMatrix.toString('bias = ')))
        // this.weights.forEach(weightMatrix => console.log(weightMatrix.toString('weight = ')))
    }

    /**
     * @param Vector a - Input to the network. Should ideally be a column
     * vector of the form [ [a1], [a2], ..., [aN] ], but the function will
     * handle row vectors [ a1, a2, ..., aN ] too.
     */
    feedforward(a) {
        if (!isMatrix(a)) {
            // Convert a to a column vector
            a = arrayToMatrix(a, 'column')
        }

        for (let i = 0; i < this.sizes.length - 1; i++) {
            console.log(`Iteration ${i+1}`)
            console.log(`${printMatrix(this.weights[i], 'w = ')}\n${printMatrix(a, 'x = ')}\n${printMatrix(this.biases[i], 'b = ')}`)
            let b = this.biases[i]
            let w = this.weights[i]
            let wa = w.multiplyMatrix(a)
            a = wa.elementWiseAdd(b).elementWiseOp(el => math.sigmoid(el))
            console.log(a.toString('a = '))
            console.log()
        }

        return jStat(a)
    }

    /**
     * @param opts.epochs - number
     * @param opts.trainingData - A list of tuples [x, y]
     * @param opts.miniBatchSize - number
     * @param opts.eta - number learning rate
     */
    sgd(opts) {
        for (let i = 0; i < opts.epochs; i++) {
            // In-place shuffle
            random.shuffle(opts.trainingData)
            let miniBatches = []
            for (let k = 0; k < opts.trainingData.length; k += opts.miniBatchSize) {
                console.log(`sgd pushing miniBatch between indices ${k}, ${k + opts.miniBatchSize}`)
                miniBatches.push(opts.trainingData.slice(k, k + opts.miniBatchSize))
            }

            miniBatches.forEach(batch => this.updateMiniBatch(batch, opts.eta))

            if (opts.testData) {
                console.log(`Epoch ${i}: ${this.evaluate(opts.testData)} / ${opts.testData.length}`)
            } else {
                console.log(`Epoch ${i} complete`)
            }
        }
    }

    /**
     * Return a tuple (nabla_b, nabla_w) representing the gradient for the
     * cost function C_x.  nabla_b and nabla_w are layer-by-layer lists of
     * matrices, similar to this.biases and this.weights.
     */
    backprop(x, y) {
        console.log('x = ', x)
        console.log('y = ', y)
        let biasPartials = this.biases.map(b => {
            let [rows, cols] = b.size()
            return Matrix.zeros(rows, cols)
        })


        let weightPartials = this.weights.map(w => {
            let [rows, cols] = w.size()
            return Matrix.zeros(rows, cols)
        })

        // Feedforward
        let activation = x
        let activations = [x] // list to store all the activations, layer by layer
        let zs = [] // list to store all the Z vectors, layer by layer
        _.zip(this.biases, this.weights).forEach((tuple) => {
            let [bias, weight] = tuple
            console.log(weight.toString('w = '))
            console.log(activation.toString('a (input) = '))
            let z = weight.multiplyMatrix(activation).elementWiseAdd(bias) // z is a vector
            console.log(z.toString('z = '))
            zs.push(z)
            activation = z.elementWiseOp(math.sigmoid)
            console.log(activation.toString('a (output) = '))
            // console.log(activation.toString('a = '))
            activations.push(activation)
        })

        // Backward pass
        let delta = this.costDerivative(activations[activations.length - 1], y).elementWiseMultiply(zs[zs.length - 1].elementWiseOp(math.sigmoidPrime))

        // biasPartials.forEach((biasPartial, i) => console.log(biasPartial.toString(`biasPartial ${i} = `)))
        // weightPartials.forEach((weightPartial, i) => console.log(weightPartial.toString(`weightPartial ${i} = `)))
        return { biasPartials, weightPartials }
    }

    /**
     * @param batch Array of [x, y] pairs where x and y are column vectors
     * (2-D arrays)
     */
    updateMiniBatch(batch, eta) {
        // Update the network's weights and biases by applying
        // gradient descent using backpropagation to a single mini batch.
        // The "miniBatch" is a list of tuples "(x, y)", and "eta"
        // is the learning rate.
        console.log('updateMiniBatch batch length', batch.length)

        // Initialize ∇b and ∇w to be vectors of zeroes. We will iterate
        // over all [x, y] pairs in the batch and average the ∂C/∂b and
        // ∂C/∂w_ij values returned from the back propagation algorithm.

        // ∇b = [ [0], [0], ..., [0] ]
        let biasPartialsSum = this.biases.map(bias => Matrix.zeros(bias.size()[0], bias.size()[1]))
        // ∇w = [ [0], [0], ..., [0] ]
        let weightPartialsSum = this.weights.map(weight => Matrix.zeros(weight.size()[0], weight.size()[1]))

        for (let i = 0; i < batch.length; i++) {
            let x = batch[i][0]
            let y = batch[i][1]
            // console.log(printMatrix(x, 'x = '))
            // console.log(printMatrix(y, 'y = '))

            let { biasPartials, weightPartials } = this.backprop(x, y)
            console.log(`${biasPartials[0].toString('∇b 0 = ')}`)

            // Add to the total ∂C/∂b accumulated so far
            _.zip(biasPartialsSum, biasPartials).forEach(tuple => {
                let [biasSum, biasPartial] = tuple
                biasSum.elementWiseAdd(biasPartial, { inplace: true })
            })

            // Add to the total ∂C/∂w_ij accumulated so far
            _.zip(weightPartialsSum, weightPartials).forEach(tuple => {
                let [weightSum, weightPartial] = tuple
                weightSum.elementWiseAdd(weightPartial, { inplace: true })
            })
        }

        biasPartialsSum.forEach((nabla, i) => console.log(nabla.toString(`accumulated ∇b ${i} = `)))
        weightPartialsSum.forEach((nabla, i) => console.log(nabla.toString(`accumulated ∇w ${i} = `)))

        // Update the biases by averaging the ∂C/∂b values from above
        // this.biases = [b-(eta/len(miniBatch))*nb for b, nb in zip(self.biases, nablaB)]
        let nablaB = biasPartialsSum.map(totalBias => totalBias.scalarMultiply(eta/batch.length))
        // nablaB.forEach((nabla, i) => console.log(nabla.toString(`averaged ∇b ${i} = `)))
        this.biases = this.biases.map((bias, i) => bias.elementWiseSubtract(nablaB[i]))

        // Update the weights by averaging the ∂C/∂w_jk values from above
        // this.weights = [w-(eta/len(miniBatch))*nw for w, nw in zip(self.weights, nablaW)]
        let nablaW = weightPartialsSum.map(totalWeight => totalWeight.scalarMultiply(eta/batch.length))
        // nablaW.forEach((nabla, i) => console.log(nabla.toString(`averaged ∇w ${i} = `)))
        this.weights = this.weights.map((weight, i) => weight.elementWiseSubtract(nablaW[i]))
    }

    /**
     * Return the number of test inputs for which the neural network outputs
     * the correct result. Note that the neural network's output is assumed
     * to be the index of whichever neuron in the final layer has the
     * highest activation.
     *
     * @param testData Array of [x, a] tuples where x is the input, and a is
     * the actual expected output
     */
    evaluate(testData) {
        // let testResults = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        let testResults = testData.map(tuple => {
            let x = tuple[0]
            let a = tuple[1]
            return [this.feedforward(x), a]
        })
        
        let totalCorrect = testResults.reduce((prev, current) => {
            let [y, a] = current
            let isEqual = _.isEqual(_.flatten(y), _.flatten(a))
            return prev += (isEqual ? 1 : 0)
        }, 0)

        return totalCorrect
    }

    /**
     * Return the vector of partial derivatives `del C_x / del a` for the
     * output activations.
     */
    costDerivative(outputActivations, y) {
        console.log(outputActivations.toString('outputActivations = '))
        console.log(y.toString('y = '))
        return outputActivations.elementWiseSubtract(y)
    }
}

let n = new Network([2, 3, 4, 2])
// let n = new Network([784, 100, 10])
// n.feedforward([[2], [3]])

n.sgd({
    epochs: 30,
    miniBatchSize: 10,
    eta: 1.0,
    trainingData: [
        [ /*x*/ Matrix.rand(2, 1), /*y*/ Matrix.rand(2, 1) ],
        [ /*x*/ Matrix.rand(2, 1), /*y*/ Matrix.rand(2, 1) ],
        [ /*x*/ Matrix.rand(2, 1), /*y*/ Matrix.rand(2, 1) ],
        [ /*x*/ Matrix.rand(2, 1), /*y*/ Matrix.rand(2, 1) ],
        [ /*x*/ Matrix.rand(2, 1), /*y*/ Matrix.rand(2, 1) ],
        [ /*x*/ Matrix.rand(2, 1), /*y*/ Matrix.rand(2, 1) ],
        [ /*x*/ Matrix.rand(2, 1), /*y*/ Matrix.rand(2, 1) ],
        [ /*x*/ Matrix.rand(2, 1), /*y*/ Matrix.rand(2, 1) ],
        [ /*x*/ Matrix.rand(2, 1), /*y*/ Matrix.rand(2, 1) ],
        [ /*x*/ Matrix.rand(2, 1), /*y*/ Matrix.rand(2, 1) ],
        [ /*x*/ Matrix.rand(2, 1), /*y*/ Matrix.rand(2, 1) ],
        [ /*x*/ Matrix.rand(2, 1), /*y*/ Matrix.rand(2, 1) ],
        [ /*x*/ Matrix.rand(2, 1), /*y*/ Matrix.rand(2, 1) ],
        [ /*x*/ Matrix.rand(2, 1), /*y*/ Matrix.rand(2, 1) ],
        [ /*x*/ Matrix.rand(2, 1), /*y*/ Matrix.rand(2, 1) ],
        [ /*x*/ Matrix.rand(2, 1), /*y*/ Matrix.rand(2, 1) ]
    ]
})

// console.log('biases')
// console.log(n.biases)
// console.log('weights')
// console.log(n.weights)
// console.log(math.sigmoid([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
