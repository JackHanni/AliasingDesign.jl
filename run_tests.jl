# Run the whole test suite and print a readable report.
#
#     julia run_tests.jl
#
# Tests waiting on a stub (`error("not implemented: f")`) are reported as pending,
# not as failures. Exits with status 1 only if a test fails or errors for real.

import Pkg
Pkg.activate(@__DIR__; io = devnull)
Pkg.resolve(; io = devnull)

using Test

# A test set that records every result and child set, so the report can walk the tree.
mutable struct ReportTestSet <: Test.AbstractTestSet
    description::String
    results::Vector{Any}
    start::Float64
    elapsed::Float64
end
ReportTestSet(desc::AbstractString; kwargs...) = ReportTestSet(String(desc), Any[], time(), 0.0)

Test.record(ts::ReportTestSet, res) = (push!(ts.results, res); res)

function Test.finish(ts::ReportTestSet)
    ts.elapsed = time() - ts.start
    Test.get_testset_depth() > 0 && Test.record(Test.get_testset(), ts)
    return ts
end

# The stub a result is waiting on, or `nothing` if it is not a stub error.
function stub_name(res)
    res isa Test.Error || return nothing
    m = match(r"not implemented: (\w+)", string(res.value))
    return m === nothing ? nothing : m.captures[1]
end

mutable struct Counts
    pass::Int
    fail::Int
    error::Int
    pending::Int
    broken::Int
end
Counts() = Counts(0, 0, 0, 0, 0)
total(c::Counts) = c.pass + c.fail + c.error + c.pending + c.broken

function tally!(c::Counts, ts::ReportTestSet, stubs, problems, path)
    for res in ts.results
        if res isa ReportTestSet
            tally!(c, res, stubs, problems, [path; res.description])
        elseif res isa Test.Pass
            c.pass += 1
        elseif res isa Test.Broken
            c.broken += 1
        elseif (name = stub_name(res)) !== nothing
            c.pending += 1
            stubs[name] = get(stubs, name, 0) + 1
        elseif res isa Test.Fail
            c.fail += 1
            push!(problems, (path, res))
        elseif res isa Test.Error
            c.error += 1
            push!(problems, (path, res))
        end
    end
    return c
end

counts(ts) = tally!(Counts(), ts, Dict{String,Int}(), [], String[])

function status(c::Counts)
    c.fail + c.error > 0 && return ("✗", :red)
    c.pending > 0 && return ("○", :yellow)
    return ("✓", :green)
end

function print_tree(ts::ReportTestSet, depth, width)
    for child in ts.results
        child isa ReportTestSet || continue
        c = counts(child)
        sym, color = status(c)
        printstyled(rpad("  "^depth * sym * " ", 2depth + 2); color)
        print(rpad(child.description, width - 2depth))
        printstyled(lpad("$(c.pass)/$(total(c))", 8); color = c.pass == total(c) ? :green : :normal)
        c.fail + c.error > 0 && printstyled("  $(c.fail + c.error) failing"; color = :red)
        c.pending > 0 && printstyled("  $(c.pending) pending"; color = :yellow)
        println()
        print_tree(child, depth + 1, width)
    end
end

function name_width(ts::ReportTestSet, depth = 0)
    w = 0
    for child in ts.results
        child isa ReportTestSet || continue
        w = max(w, 2depth + length(child.description), name_width(child, depth + 1))
    end
    return w
end

root = @testset ReportTestSet "root" begin
    include(joinpath(@__DIR__, "test", "runtests.jl"))
end

stubs = Dict{String,Int}()
problems = []
c = tally!(Counts(), root, stubs, problems, String[])
width = name_width(root) + 2
rule = "─"^(width + 30)

println()
printstyled("AliasingDesign.jl test report\n"; bold = true)
println(rule)
print_tree(root, 0, width)
println(rule)
printstyled("Passed $(c.pass)"; color = :green)
print(" · ")
printstyled("Failed $(c.fail)"; color = c.fail > 0 ? :red : :normal)
print(" · ")
printstyled("Errored $(c.error)"; color = c.error > 0 ? :red : :normal)
print(" · ")
printstyled("Pending $(c.pending)"; color = c.pending > 0 ? :yellow : :normal)
c.broken > 0 && print(" · Broken $(c.broken)")
println(" · Total $(total(c))   ($(round(root.elapsed; digits = 1)) s)")

if !isempty(stubs)
    println()
    printstyled("Not implemented yet (tests waiting on each):\n"; color = :yellow, bold = true)
    for (name, n) in sort(collect(stubs); by = last, rev = true)
        println("  ", rpad(name, 24), n)
    end
end

relpath_(s) = replace(string(s), joinpath(@__DIR__, "") => "", Base.contractuser(@__DIR__) * "/" => "")
location(res) = relpath_("$(res.source.file):$(res.source.line)")

# One-line cause of a failure, and where in src/ it came from (for errors).
function cause(res)
    if res isa Test.Fail
        evaluated = res.data === nothing ? "" : "  (evaluated: $(res.data))"
        return ("Failed: $(res.orig_expr)$evaluated", "")
    end
    # The backtrace text starts with the readable error message (as `showerror` prints it).
    message = strip(first(split(string(res.backtrace), '\n')))
    m = match(r"@ AliasingDesign (\S+)", string(res.backtrace))
    return (message, m === nothing ? "" : "at " * relpath_(m.captures[1]))
end

if !isempty(problems)
    println()
    printstyled("Failures and errors:\n"; color = :red, bold = true)
    if "--verbose" in ARGS
        for (i, (path, res)) in enumerate(problems)
            println()
            printstyled("[$i] ", join(path, " › "), "\n"; color = :red, bold = true)
            show(stdout, MIME"text/plain"(), res)
            println()
        end
    else
        # Group tests that fail for the same reason, so one bug prints once.
        groups = Dict{Tuple{String,String},Vector{Any}}()
        order = Tuple{String,String}[]
        for (path, res) in problems
            key = cause(res)
            haskey(groups, key) || push!(order, key)
            push!(get!(groups, key, []), (path, res))
        end
        for (i, key) in enumerate(order)
            println()
            printstyled("[$i] ", key[1], "\n"; color = :red, bold = true)
            isempty(key[2]) || println("    ", key[2])
            members = groups[key]
            println("    $(length(members)) test$(length(members) == 1 ? "" : "s"):")
            for (path, res) in members
                println("      ", rpad(location(res), 34), join(path[2:end], " › "))
            end
        end
        println()
        println("Run `julia run_tests.jl --verbose` for full stack traces.")
    end
end
println()

exit(c.fail + c.error > 0 ? 1 : 0)
