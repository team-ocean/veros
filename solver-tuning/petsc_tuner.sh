
{
ksp_type="-ksp_type";
pc_type="-pc_type";
ksp_types=("bcgs" "cg" "gmres");
pc_types=("pfmg" "gamg")
setup="--size 100 100 100 --timesteps 5 --device gpu -ls petsc";
}
for i in ${pc_types[@]};
do
echo python ~/veros/benchmarks/streamfunction_solver_benchmark.py ${setup} --petsc-options ${pc_type}\ ${i}
python ~/veros/benchmarks/streamfunction_solver_benchmark.py ${setup} --petsc-options ${pc_type}\ ${i}
done