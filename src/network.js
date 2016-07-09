'use strict'

let jStat = require('jstat').jStat
let sprintf = require('sprintf').sprintf
let random = require('./random')
let math = require('./math')

function printMatrix(m, label) {
    if (m instanceof jStat) {
        m = Array.prototype.slice.call(m)
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
        return arr.map(e => [e])
    } else if (type === 'row') {
        return [arr]
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
        this.biases = this.sizes.slice(1).map(layerSize => jStat.rand(layerSize, 1))

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
            return jStat.rand(yLayerSize, xLayerSize)
        })
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
            let wa = jStat.multiply(w, a)
            a = jStat(wa).add(b).alter(el => math.sigmoid(el))
            console.log(printMatrix(a, 'a = '))
            console.log()
        }

        return a
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

    backprop(x, y) {
        let deltaNablaB = 1
        let deltaNablaW = 2
        return { deltaNablaB, deltaNablaW }
    }

    /**
     * @param batch Array of [x, y] pairs
     */
    updateMiniBatch(batch, eta) {
        // Update the network's weights and biases by applying
        // gradient descent using backpropagation to a single mini batch.
        // The "miniBatch" is a list of tuples "(x, y)", and "eta"
        // is the learning rate.
        console.log('updateMiniBatch batch length', batch.length)

        //           [np.zeros(b.shape) for b in self.biases]
        let nablaB = this.biases.map(bias => jStat.zeros(bias.length, bias[0].length))
        //           [np.zeros(w.shape) for w in self.weights]
        let nablaW = this.weights.map(weight => jStat.zeros(weight.length, weight[0].length))

        for (let i = 0; i < batch.length; i++) {
            let x = batch[i][0]
            let y = batch[i][1]
            // console.log(printMatrix(x, 'x = '))
            // console.log(printMatrix(y, 'y = '))

            let {deltaNablaB, deltaNablaW} = this.backprop(x, y)
            //       [nb+dnb for nb, dnb in zip(nablaB, deltaNablaB)]
            nablaB = deltaNablaB.map((_, i) => nablaB[i] + deltaNablaB[i])
            // nablaW = [nw+dnw for nw, dnw in zip(nablaW, deltaNablaW)]
            nablaW = deltaNablaW.map((_, i) => nablaW[i] + deltaNablaW[i])
        }

        // this.weights = [w-(eta/len(miniBatch))*nw for w, nw in zip(self.weights, nablaW)]
        this.weights = this.weights.map((_, i) => this.weights[i]-(opts.eta/miniBatch.length)*nablaW[i])
        // this.biases = [b-(eta/len(miniBatch))*nb for b, nb in zip(self.biases, nablaB)]
        this.biases = this.biases.map((_, i) => this.biases[i]-(opts.eta/lminiBatch.length)*nablaB[i])
    }
}

let n = new Network([2, 3, 4, 2])
// n.feedforward([[2], [3]])

n.sgd({
    epochs: 3,
    miniBatchSize: 5,
    eta: 0.9,
    trainingData: [
        [ /*x*/ jStat.rand(100, 1), /*y*/ random.getRandomOutputVector(10) ],
        [ /*x*/ jStat.rand(100, 1), /*y*/ random.getRandomOutputVector(10) ],
        [ /*x*/ jStat.rand(100, 1), /*y*/ random.getRandomOutputVector(10) ],
        [ /*x*/ jStat.rand(100, 1), /*y*/ random.getRandomOutputVector(10) ],
        [ /*x*/ jStat.rand(100, 1), /*y*/ random.getRandomOutputVector(10) ],
        [ /*x*/ jStat.rand(100, 1), /*y*/ random.getRandomOutputVector(10) ],
        [ /*x*/ jStat.rand(100, 1), /*y*/ random.getRandomOutputVector(10) ],
        [ /*x*/ jStat.rand(100, 1), /*y*/ random.getRandomOutputVector(10) ],
        [ /*x*/ jStat.rand(100, 1), /*y*/ random.getRandomOutputVector(10) ],
        [ /*x*/ jStat.rand(100, 1), /*y*/ random.getRandomOutputVector(10) ],
        [ /*x*/ jStat.rand(100, 1), /*y*/ random.getRandomOutputVector(10) ],
        [ /*x*/ jStat.rand(100, 1), /*y*/ random.getRandomOutputVector(10) ],
        [ /*x*/ jStat.rand(100, 1), /*y*/ random.getRandomOutputVector(10) ],
        [ /*x*/ jStat.rand(100, 1), /*y*/ random.getRandomOutputVector(10) ],
        [ /*x*/ jStat.rand(100, 1), /*y*/ random.getRandomOutputVector(10) ],
        [ /*x*/ jStat.rand(100, 1), /*y*/ random.getRandomOutputVector(10) ]
    ]
})

// console.log('biases')
// console.log(n.biases)
// console.log('weights')
// console.log(n.weights)
// console.log(math.sigmoid([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
