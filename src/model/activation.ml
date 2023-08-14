open Torch

let gelu_new =
  let sqrt_two_over_pi = 2. /. Float.pi |> Float.sqrt in
  fun xs ->
    let ys =
      Tensor.(
        tanh (((pow xs ~exponent:(Tensor.of_float0 3.) * f 0.044715) + xs) * f sqrt_two_over_pi))
    in
    Tensor.(xs * f 0.5 * (ys + f 1.))

let gelu xs =
  let erf = Tensor.erf (Tensor.div xs (Tensor.of_float0 (Float.sqrt 2.))) in
  Tensor.mul (Tensor.add erf (Tensor.of_float0 1.)) (Tensor.of_float0 0.5) |> Tensor.mul xs

let relu = Tensor.relu
let mish xs = Tensor.(xs * tanh (softplus xs))
let swish xs = Tensor.(xs * sigmoid xs)
