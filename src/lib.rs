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
    usize: AsPrimitive<S>,
    StandardNormal: Distribution<S>,
    R: RngCore,
    F1: Fn(&Codomain, &Codomain) -> std::cmp::Ordering,
    F2: Fn(&Codomain) -> bool,
{
    domain_dimension: usize,
    samples: Vec<Sample<Domain<S>, Codomain>>,
    population_size: usize, //aka. lambda
    rng: R,
    eigen_vectors_ortho: Matrix<S>, //aka. B
    eigen_values_sqrt: Domain<S>,   //diagonal elements of D
    comparison_function: F1,
    feasibility_function: F2,
    weights: VectorN<S, Dynamic>,
    mu: S,     // /!\ should be updated along with weights
    mu_eff: S, // /!\ should be updated along with weights
    //sigma
    step_size: S,
    mean: Domain<S>,
    conjugate_evolution_path: Domain<S>, //aka p_sigma
    c_sigma: S,
    d_sigma: S,
}

impl<S, R, Codomain, F1, F2> Step<S, R, Codomain, F1, F2>
where
    S: DomainScalar,
    usize: AsPrimitive<S>,
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
        // <y>_w in the paper
        let weighted_mean_displacement = {
            assert!(displacements.len() >= 2);
            utils::weighted_sum(&displacements, &self.weights)
        };
        let new_mean = utils::weighted_sum(self.samples.iter().map(|s| &s.point), &self.weights);
        let new_conjugate_evolution_path = {
            let mut p_s = self.conjugate_evolution_path;
            let two = S::one() + S::one();
            // sqrt(cσ(2−cσ)μeff)
            let norm_cst = (self.c_sigma * (two - self.c_sigma) * self.mu_eff).sqrt();
            // C^{-1/2}*<y>_w
            let tmp = {
                let mut tmp = weighted_mean_displacement.clone();
                tmp.zip_apply(&self.eigen_values_sqrt, |x, y| x / y);
                tmp
            };
            // p_s = alpha * a * x + beta * p_s
            p_s.gemv(
                norm_cst,                  //alpha
                &self.eigen_vectors_ortho, //a
                &tmp,
                S::one() - self.c_sigma, // beta
            );
            p_s
        };
        let new_step_size = {
            let two = S::one() + S::one();
            let four = two + two;
            let twenty_one = four + four + four + four + four + S::one();
            let n: S = self.domain_dimension.as_();
            let expected_standard_normal_norm_approx =
                n.sqrt() * (S::one() - (four * n).recip() + (twenty_one * n * n).recip());
            self.step_size
                * (self.c_sigma / self.d_sigma * new_conjugate_evolution_path.norm()
                    / expected_standard_normal_norm_approx)
        };
        unimplemented!()
    }
}
