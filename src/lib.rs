use nalgebra::*;
use num_traits::AsPrimitive;
use rand::prelude::*;
use rand_distr::*;

mod utils;

const STD_NORMAL_DISTRIBUTION: StandardNormal = StandardNormal {};

pub trait DomainScalar: nalgebra::RealField {}

type Domain<S> = VectorN<S, Dynamic>;
type Matrix<S> = MatrixN<S, Dynamic>;

// struct Strategy<S>
// {
//     domain_dimension : usize,

// }

struct Sample<P, V> {
    point: P,
    value: V,
}

pub struct Step<S, R, Codomain, F1, F2>
where
    S: DomainScalar,
    S: std::iter::Sum,
    usize: AsPrimitive<S>,
    f32: AsPrimitive<S>,
    StandardNormal: Distribution<S>,
    R: RngCore,
    F1: Fn(&Codomain, &Codomain) -> std::cmp::Ordering,
    F2: Fn(&Codomain) -> bool,
{
    domain_dimension: usize,
    samples: Vec<Sample<Domain<S>, Codomain>>,
    population_size: usize, //aka. lambda
    rng: R,
    covariance_matrix: Matrix<S>,
    eigen_vectors_ortho: Matrix<S>, //aka. B
    eigen_values_sqrt: Domain<S>,   //diagonal elements of D
    comparison_function: F1,
    feasibility_function: F2,
    weights: VectorN<S, Dynamic>,
    mu: usize, // /!\ should be updated along with weights
    mu_eff: S, // /!\ should be updated along with weights
    //sigma
    step_size: S,
    mean: Domain<S>,
    conjugate_evolution_path: Domain<S>, //aka p_sigma
    c_sigma: S,
    d_sigma: S,
    evolution_path: Domain<S>,
    c_c: S,
    generation_number: u32,
    c_one: S,
    c_mu: S,
}

impl<S, R, Codomain, F1, F2> Step<S, R, Codomain, F1, F2>
where
    S: DomainScalar,
    S: std::iter::Sum,
    usize: AsPrimitive<S>,
    f32: AsPrimitive<S>,
    StandardNormal: Distribution<S>,
    R: RngCore,
    F1: Fn(&Codomain, &Codomain) -> std::cmp::Ordering,
    F2: Fn(&Codomain) -> bool,
{
    pub fn samples_still_needed(&self) -> bool {
        self.samples.len() < self.population_size
    }

    pub fn ask(&mut self) -> Domain<S> {
        let n: usize = self.domain_dimension;
        // Generate the normalized random sample
        let mut z: Domain<S> =
            Domain::<S>::from_distribution(n, &STD_NORMAL_DISTRIBUTION, &mut self.rng);
        // Multiply it by the diagonal matrix of the square roots of the eigenvalues
        // ie element-wise with self.eigen_values_sqrt.
        // This is done in place to save an alloc
        z.zip_apply(&self.eigen_values_sqrt, |x, y| x * y);
        // And multiply by the eigen vectors
        let mut y = &self.eigen_vectors_ortho * z;
        y *= self.step_size;
        y += &self.mean;
        // y has become x inplace
        let x = y;
        x
    }

    pub fn tell(&mut self, point: Domain<S>, value: Codomain) -> Result<(), ()> {
        if (self.feasibility_function)(&value) {
            self.samples.push(Sample {
                point: point,
                value: value,
            });
            Ok(())
        } else {
            Err(())
        }
    }

    pub fn tell_unchecked(&mut self, point: Domain<S>, value: Codomain) {
        self.samples.push(Sample {
            point: point,
            value: value,
        });
    }

    pub fn next_step(mut self) -> Self {
        // Following page 29 of
        // The CMA Evolution Strategy: A Tutorial
        // arXiv:1604.00772 (version 1)
        // https://arxiv.org/pdf/1604.00772v1.pdf
        if self.samples_still_needed() {
            panic!("Not enough samples!")
        }

        // Order the samples (x_i), the best first and worst last
        // Affecting to variables beforehand is necessary to appease
        // our Lord the borrow-checker.
        {
            let samples = &mut self.samples;
            let f = &self.comparison_function;
            samples.sort_unstable_by(|x, y| f(&x.value, &y.value));
        }
        // "y_i"s in the paper
        let displacements: Vec<Domain<S>> = self
            .samples
            .iter()
            .map(|x| (&x.point - &self.mean) / self.step_size)
            .collect();

        // " Selection and recombination"

        // <y>_w in the paper
        let weighted_mean_displacement =
            { utils::weighted_sum(&displacements[..self.mu], &self.weights) };
        let new_mean = utils::weighted_sum(self.samples.iter().map(|s| &s.point), &self.weights);

        let mut new_samples = self.samples;
        new_samples.clear();

        // Constants used afterwards
        let one = 1.as_();
        let two = 2.as_();
        let four = 4.as_();
        let twenty_one = 21.as_();
        let n = self.domain_dimension.as_();
        //sqrt(n) * (1- 1/42 + 1/21n²)
        let expected_standard_normal_norm_approx =
            n.sqrt() * (one - (four * n).recip() + (twenty_one * n * n).recip());

        // C = BDB'
        let cov_max_sqrt = {
            let mut bd = self.eigen_vectors_ortho.clone();
            for (mut ci, &di) in bd.column_iter_mut().zip(self.eigen_values_sqrt.iter()) {
                ci *= di;
            }
            bd * self.eigen_vectors_ortho.transpose()
        };

        // "Step-size control"
        let new_conjugate_evolution_path: Domain<S> = {
            let mut p_s = self.conjugate_evolution_path;
            // sqrt(cσ(2−cσ)μeff)
            let norm_cst = (self.c_sigma * (two - self.c_sigma) * self.mu_eff).sqrt();

            // p_s = alpha * a * x + beta * p_s
            p_s.gemv(
                norm_cst,                    //alpha
                &cov_max_sqrt,               //a
                &weighted_mean_displacement, //x
                one - self.c_sigma,          // beta
            );
            p_s
        };
        let new_conjugate_evolution_path_norm = new_conjugate_evolution_path.norm();
        let new_step_size = {
            self.step_size
                * (self.c_sigma / self.d_sigma * new_conjugate_evolution_path_norm
                    / expected_standard_normal_norm_approx)
        };

        // "Covariance matrix adaptation"
        let p_sigma_too_large = {
            // g + 2 because g has not been incremented yet
            let lhs = new_conjugate_evolution_path_norm
                / (one - (one - self.c_sigma).powi(2 * (self.generation_number + 2) as i32)).sqrt();
            let rhs = ((1.4_f32).as_() + two / (n + one)) * expected_standard_normal_norm_approx;
            lhs >= rhs
        };
        let new_evolution_path = {
            // hσ * sqrt(c_c (2−c_c) μeff)
            let mut p_c = self.evolution_path;
            p_c *= one - self.c_c;
            if !p_sigma_too_large {
                let norm_cst = (self.c_c * (two - self.c_c) * self.mu_eff).sqrt();
                p_c.axpy(norm_cst, &weighted_mean_displacement, one)
            }
            p_c
        };
        // ∑wj
        // Compute it now to reuse the buffer
        let recomb_weight_sum = self.weights.sum();

        for (w, y) in self.weights.iter_mut().zip(&displacements) {
            if *w < S::zero() {
                let tmp = &cov_max_sqrt * y;
                *w *= n / tmp.norm_squared();
            }
        }

        let new_cov_matrix = {
            let mut res = self.covariance_matrix;
            let mut acc = new_conjugate_evolution_path.kronecker(&new_conjugate_evolution_path);
            acc.scale_mut(self.c_one);
            for (&w, y) in self.weights.iter().zip(&displacements) {
                let tmp = y.kronecker(&y);
                acc.axpy(self.c_mu * w, &tmp, one)
            }
            let mul_const = if p_sigma_too_large {
                one + self.c_one * self.c_c * (two - self.c_c)
                    - self.c_one
                    - self.c_mu * recomb_weight_sum
            } else {
                one - self.c_one - self.c_mu * recomb_weight_sum
            };
            res.scale_mut(mul_const);
            res += acc;
            res
        };
        let eigens = new_cov_matrix.clone().symmetric_eigen();
        let mut new_eigenvalues_sqrt = eigens.eigenvalues;
        for l in new_eigenvalues_sqrt.iter_mut() {
            *l = l.sqrt();
        }
        assert!(eigens.eigenvectors.is_orthogonal(S::default_epsilon()));

        Self {
            covariance_matrix: new_cov_matrix,
            samples: new_samples,
            eigen_vectors_ortho: eigens.eigenvectors,
            eigen_values_sqrt: new_eigenvalues_sqrt,
            mu_eff: self.weights.iter().map(|&x| x * x).sum::<S>().recip(),
            step_size: new_step_size,
            mean: new_mean,
            conjugate_evolution_path: new_conjugate_evolution_path,
            evolution_path: new_evolution_path,
            generation_number: self.generation_number + 1,
            ..self
        }
    }
}
