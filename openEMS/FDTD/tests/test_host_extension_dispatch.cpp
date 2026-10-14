// Exercise the production dispatcher independently of the solver libraries.
// Small engine stand-ins make the selected accessor type observable. This is
// a host dispatch regression, not a Metal simulation or a field-accuracy test.
#include <iostream>
#include <stdexcept>
#include <type_traits>

// Suppress only the heavyweight engine declarations pulled in by the dispatcher.
#define ENGINE_H
#define ENGINE_SSE_H
class Engine {
public:
    enum EngineType { BASIC, SSE, CUDA, METAL, UNKNOWN };
    explicit Engine(EngineType type) : type_(type) {}
    EngineType GetType() const { return type_; }
private:
    EngineType type_;
};
class Engine_sse : public Engine {
public:
    Engine_sse() : Engine(SSE) {}
};

#include "FDTD/extensions/engine_extension_dispatcher.h"

class Probe {
public:
    explicit Probe(Engine* engine) : m_Eng(engine), calls(0), host(false), value(0) {}
    void dispatch() { ENG_DISPATCH(apply); }
    void dispatch_args(int arg) { ENG_DISPATCH_ARGS(apply_args, arg); }
    template <class T> void apply(T*) {
        ++calls;
        host = std::is_same<T, Engine>::value;
    }
    template <class T> void apply_args(T* engine, int arg) {
        apply(engine);
        value = arg;
    }
    Engine* m_Eng;
    int calls;
    bool host;
    int value;
};

void check(Engine& engine, bool expected_host) {
    Probe p(&engine);
    p.dispatch();
    if (p.calls != 1 || p.host != expected_host)
        throw std::runtime_error("Wrong no-argument extension dispatch");
    p.dispatch_args(37);
    if (p.calls != 2 || p.host != expected_host || p.value != 37)
        throw std::runtime_error("Wrong argument extension dispatch");
}

int main() {
    try {
        Engine basic(Engine::BASIC), metal(Engine::METAL);
        Engine_sse sse;
        check(basic, true);
        check(sse, false);
        check(metal, true);
        for (auto tag : {Engine::CUDA, Engine::UNKNOWN}) {
            Engine unsupported(tag);
            Probe p(&unsupported);
            bool rejected = false;
            try { p.dispatch(); } catch (const std::runtime_error&) { rejected = true; }
            if (!rejected) throw std::runtime_error("Unsupported engine accepted");
            rejected = false;
            try { p.dispatch_args(37); } catch (const std::runtime_error&) { rejected = true; }
            if (!rejected || p.calls) throw std::runtime_error("Unsupported engine callback ran");
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    std::cout << "BASIC/SSE/METAL dispatch passed; CUDA/UNKNOWN rejected\n";
    return 0;
}
