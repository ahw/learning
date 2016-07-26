'use strict'

let sprintf = require('sprintf').sprintf

class Matrix {
    constructor(rows, cols, fn) {
        if (typeof rows === 'object') {
            // Passed in literal data
            this._data = rows
            return this
        }

        fn = fn || (() => 0)
        this._data = Array.from({ length: rows }, (_, i) => {
            return Array.from({ length: cols }, (_, j) => {
                return fn(i, j)
            })
        })

        return this
    }

    size() {
        return [this._data.length, this._data[0].length]
    }

    elementWiseOp(fn, opts) {
        let options = Object.assign({
            inplace: true
        }, opts)

        let inputMatrix = this
        let returnMatrix = undefined
        if (options.inplace) {
            returnMatrix = this
        } else {
            returnMatrix = new Matrix(this._data.length, this._data[0].length)
        }

        for (let i = 0; i < returnMatrix._data.length; i++) {
            for (let j = 0; j < returnMatrix._data[i].length; j++) {
                returnMatrix._data[i][j] = fn(inputMatrix._data[i][j], i, j)
            }
        }
        
        return returnMatrix
    }

    scalarAdd(n, options) {
        return this.elementWiseOp((e) => e + n, options)
    }

    scalarSubtract(n, options) {
        return this.elementWiseOp((e) => e - n, options)
    }

    scalarMultiply(n, options) {
        return this.elementWiseOp((e) => e * n, options)
    }

    scalarDivide(n, options) {
        return this.elementWiseOp((e) => e / n, options)
    }

    addMatrix(matrix, options) {
        return this.elementWiseOp((value, i, j) => {
            return value + matrix._data[i][j]
        }, options)
    }

    multiplyMatrix(matrix, options) {
        // MxN * NxP = MxP
        let result = Matrix.create(this._data.length, matrix._data[0].length)
        return result.elementWiseOp((value, i, j) => {
            let row = this._data[i]
            let col = matrix._data.map(row => row[j])
            return row.reduce((prev, curr, i) => prev + row[i]*col[i], 0)
        })
    }

    elementWiseAdd(matrix, options) {
        return this.elementWiseOp((value, i, j) => {
            return value + matrix._data[i][j]
        }, options)
    }

    elementWiseSubtract(matrix, options) {
        return this.elementWiseOp((value, i, j) => {
            return value - matrix._data[i][j]
        }, options)
    }


    elementWiseMultiply(matrix, options) {
        return this.elementWiseOp((value, i, j) => {
            return value * matrix._data[i][j]
        }, options)
    }

    toString(label) {
        let matrix = this._data
        label = label || ""
        let pad = label.split("").map(s => ' ').join("")
        return matrix.map((row, i) => {
            if (!Array.isArray(row)) {
                return `${(i ? pad : label)} ${row} <-- warning: this is not a matrix`
            }

            return (i ? pad : label) + '[ ' + row.map(e => sprintf('%0.03f', e)).join(', ') + ' ]'
        }).join('\n') + '\n'
    }
}

Matrix.create = function(rows, cols, fn) {
    return new Matrix(rows, cols, fn)
}

Matrix.reverseCreate = function(fn, rows, cols) {
    return new Matrix(rows, cols, fn)
}

Matrix.ones = Matrix.reverseCreate.bind(null, () => 1)
Matrix.zeros = Matrix.reverseCreate.bind(null, () => 0)
Matrix.rand = Matrix.reverseCreate.bind(null, () => Math.random())
Matrix.randInt = Matrix.reverseCreate.bind(null, () => Math.floor(Math.random() * 10))

module.exports = Matrix
// console.log(Matrix.create(4, 1).toString())
// console.log(Matrix.ones(4, 1).toString())
// console.log(Matrix.zeros(4, 1).toString())
// let r = Matrix.rand(7, 2)
// console.log(r)
// let s = r.scalarAdd(1, { inplace: false })
// console.log(r)
// console.log(s)

// let twos = Matrix.ones(7, 2).scalarMultiply(2)
// console.log(twos)

// console.log(twos.addMatrix(s, { inplace: false }))
// console.log(twos)

// let f = Matrix.create(4, 3, (i, j) => i + j)
// let v = Matrix.create(3, 1, (i, j) => i + 1)
// console.log(f)
// console.log(v)
// let result = f.multiplyMatrix(v)
// console.log(result)