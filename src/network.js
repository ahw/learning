'use strict'

let jStat = require('jstat').jStat
let sprintf = require('sprintf').sprintf

function sigmoid(z) {
    if (typeof z === 'number') {
        return 1/(1+Math.pow(Math.E, -z))
    } else {
        return z.map(sigmoid)
    }
}

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
            a = jStat(wa).add(b).alter(el => sigmoid(el))
            console.log(printMatrix(a, 'a = '))
            console.log()
        }

        return a
    }
}

let n = new Network([2, 3, 4, 2])
n.feedforward([[2], [3]])
// console.log('biases')
// console.log(n.biases)
// console.log('weights')
// console.log(n.weights)
// console.log(sigmoid([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
