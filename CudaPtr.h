#pragma once
#include <cuda_runtime.h>

template <typename T>
class CudaUniquePtr
{
public:
    explicit CudaUniquePtr(T* ptr = nullptr) : m_ptr(ptr) {}

    // Удаление конструктора копирования и оператора присваивания копированием
    CudaUniquePtr(CudaUniquePtr<T> const& other) = delete;
    CudaUniquePtr<T>& operator=(CudaUniquePtr<T> const& other) = delete;

    // Реализация перемещающего конструктора
    CudaUniquePtr(CudaUniquePtr<T>&& other) noexcept : m_ptr(other.m_ptr)
    {
        other.m_ptr = nullptr;
    }

    // Реализация перемещающего оператора присваивания
    CudaUniquePtr<T>& operator=(CudaUniquePtr<T>&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            m_ptr = other.m_ptr;
            other.m_ptr = nullptr;
        }
        return *this;
    }

    T** operator& ()
    {
        reset();
        return &m_ptr;
    }

    T const* get() const { return m_ptr; }
    T* get() { return m_ptr; }

    void reset(T* ptr = nullptr)
    {
        if (m_ptr)
        {
            cudaFree(m_ptr);
        }
        m_ptr = ptr;
    }

    ~CudaUniquePtr()
    {
        reset();
    }

private:
    T* m_ptr;
};
