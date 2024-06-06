#pragma once
#include <cstddef>
#include <cstdint>

class Matrix {
public:
    enum MulMode {
        SIMPLE,
        SHARED,
        INTRINSICS
    };

    Matrix(size_t h, size_t w);
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    static Matrix full(float val, size_t h, size_t w);
    static Matrix rand(size_t h, size_t w);
    Matrix mul(const Matrix& other, MulMode mode) const;
    float at(size_t i, size_t j) const;
    float& at(size_t i, size_t j);
    size_t height() const;
    size_t width() const;

    virtual ~Matrix();

private:
    float* m_data;
    size_t m_h, m_w;
};
