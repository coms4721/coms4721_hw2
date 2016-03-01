## Problem 3
$ \sum v = \lambda v $

####a) eigenvalues of $ \sum  + \sigma^2 I $

$ \sum = \begin{bmatrix} 
\sigma_1^2 & \sigma_{12} \\
\sigma_{21} & \sigma_2^2 
\end{bmatrix}$

$ \sigma^2 I + \sum = \begin{bmatrix} 
\sigma_1^2 & \sigma_{12} \\
\sigma_{21} & \sigma_2^2 
\end{bmatrix} + \begin{bmatrix} 
\sigma^2 & 0 \\
0 & \sigma^2 
\end{bmatrix}$

$ \sum \leftarrow orginal \ covariance $

$ \sigma^2 \begin{bmatrix} 
1 & 0 \\
0 & 1 
\end{bmatrix} = \begin{bmatrix} 
\sigma^2 & 0 \\
0 & \sigma^2 
\end{bmatrix}$

$ \sum + \sigma^2I =  \begin{bmatrix} 
\sigma_1^2 + \sigma^2 & \sigma_{12} \\
\sigma_{21} & \sigma_2^2 + \sigma^2
\end{bmatrix} $

---
$ ( \sum + \sigma^2I) v' = \lambda'v'$

---

$ \sum v' + \sigma^2 I v' = \lambda'v' $

$ \sum v' = \lambda'v' - \sigma^2Iv' $

---
$ \sum v' = (\lambda' - \sigma^2)v' $

---

$ v' \ eigenvector \ of \sum $

$ x' - \sigma^2 \ eigenvalue \ of \sum $

$ \lambda' - \sigma^2 = \lambda $

---
$ \lambda' = \lambda + \sigma^2 $

---

####b) eigenvalues of $ (\sum  + \sigma^2 I)^{-1} $

$ ( \sum + \sigma^2I )^{-1} v' = \lambda'v'        $

$ v' = \lambda' ( \sum + \sigma^2 I ) v' $ 

$ \dfrac{1}{\lambda'} v' = ( \sum + \sigma^2 I ) v' $

$ = \sum v' + \sigma^2 v' $

$ \sum v' = \dfrac{1}{\lambda'} v'  - \sigma^2 v' $

$ \sum v' = (\dfrac{1}{\lambda'}  - \sigma^2 ) v' , \leftarrow this \ here \ is \ \lambda$

$ \lambda = \dfrac{1}{\lambda'} - \sigma^2 $

$ \lambda + \sigma^2 = \dfrac{1}{\lambda'} $

---
$ \lambda' = \dfrac{1}{\lambda + \sigma^2} $

---

####c) 

$ RR = \sum + \sigma^2 I $

$ R^{-1}R = I, \ \ RR^{-1} = I $

$ \sum v_1 = \lambda_1 v_1 $ 

$ E(X) = 0 $

$ Y = v_1^T R^{-1} X $

$ \mu_Y = E(Y) = E(v_1^T, R^{-1} X) $

$ = v_1^T R^{-1} E(X), \leftarrow which \ is \ 0 $

$ Var(Y) = E[Y^2] $

$ = E [(v_1^T R^{-1} X) ( v_1^T R^{-1} X)^T ] $

$ = E [v_1^TR^{-1} X X^T R^{-T} v_1 ]    $


$ = v_1^TR^{-1} E( X X^T)  R^{-T} v_1     $

$ = v_1^T R^{-1} \sum R^{-T} v_1 $

$ RR = \sum + \sigma^2I, \ \ \ \ \ \  R = R^T \rightarrow R^{-1} = (R^{-1})^T $

$R^{-1}RR = R^{-1} \sum + \sigma^2 R^{-1} I $

^ $ \ above \ is \ IR, \ which \ is \ R $

$ R = R^{-1}\sum + \sigma^2 R^{-1}, \ \ \ x(R^{-1})^T = R^{-1}$

$ RR^{-1} = R^{-1} \sum R^{-T} + \sigma^2 R^{-1} R^{-1} $

$ I = R^{-1} \sum R^{-T} + \sigma^2 R^{-1} R^{-1} $

$R^{-1} \sum R^{-T} = I - \sigma^2 R{-1} R^{-1} $

$Var(Y) = v_1^T(I - \sigma^2 R^{-1} R^{-1} ) v_1 $

$ = v_1^T v_1 - \sigma^2 v_1^T R^{-1} R^{-1}  v_1 $

$ = 1 - \sigma^2 v_1^T R^{-1} R^{-1} v_1 $

$ \sum v_1 = \lambda_1 v_1, \leftarrow \sum = RR - \sigma^2I $ 

$ RR = \sum + \sigma^2 I $

$ RR v_1 = (\lambda_1 + \sigma^2 ) v_1 $
multiply both sides by $R^{-1}$

$ R^{-1}RR v_1 = R^{-1}(\lambda_1 + \sigma^2 ) v_1 $

^ $ is \ I $

$ Rv_1  = (\lambda_1 + \sigma^2) R^{-1} v_1 $

---
$ R^{-1}v_1 = \dfrac{R v_1}{\lambda_1 + \sigma^2} $

---

transpose

$ (R^{-1} v_1)^T = V_1^T R^{-T} = v_1^T R^{-1} = \dfrac{v_1^T R^T}{\lambda_1 + \sigma^2} $

---
$ \rightarrow v_1^T R^{-1} = \dfrac{v_1^T R}{\lambda_1 + \sigma^2} $

---

$ var(Y) = 1 - \sigma^2 v_1^T R^{-1} R^{-1} v_1 $

$ = 1 - \sigma^2 (\dfrac{v_1^T R}{\lambda_1 + \sigma^2})(\dfrac{Rv_1}{\lambda_1 + \sigma^2}) $

$ = 1 - \dfrac{\sigma^2}{(\lambda_1 + \sigma^2)^2} v_1^T ( RR ) v_1  $

$ \ \ \ \  note: RR = \sum + \sigma^2 I $

$ = 1 - \dfrac{\sigma^2}{(\lambda_1 + \sigma^2)^2} [v_1^T \sum v_1 + \sigma^2 v_1^T v_1$

$ \ \ \ v_1^Tv_1 \ is \ 1 $

$ = 1 - \dfrac{\sigma^2}{(\lambda_1 + \sigma^2)^2} (\lambda_1 + \sigma^2)    $

---
$ = 1 - \dfrac{\sigma^2}{\lambda_1 + \sigma^2}  $

---