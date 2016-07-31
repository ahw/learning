'use strict'

let path = require('path')
const webpack = require('webpack')

module.exports = {
    entry: './src/app.jsx',
    output: {
        path: path.resolve(__dirname, 'dist'),
        filename: 'app.js',
        // publicPath: 'https://localhost:8082/'
        publicPath: '/xx-assets-xx/'
    },
    module: {
        loaders: [{
            test: /\.jsx?$/,
            exclude: /node_modules/,
            loader: 'babel-loader'
        }]
    },
    devtool: 'source-map'
    // Instead of hard-coding this Uglify plugin configuration, you can just
    // run webpack with the -p or --optimize-minimize flag which will minify
    // stuff.
    // plugins: [
    //     // Turn on for production minification
    //     // new webpack.optimize.UglifyJsPlugin({
    //     //     compress: {
    //     //         warnings: false,
    //     //     },
    //     //     output: {
    //     //         comments: false,
    //     //     }
    //     // })
    // ]
 }
