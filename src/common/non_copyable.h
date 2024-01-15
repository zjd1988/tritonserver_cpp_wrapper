/********************************************
 * @Author: zjd
 * @Date: 2024-01-11 
 * @LastEditTime: 2024-01-11 
 * @LastEditors: zjd
 ********************************************/
#pragma once

namespace TRITON_SERVER
{
    
    /** protocol class. used to delete assignment operator. */
    class NonCopyable
    {
    public:
        NonCopyable()                    = default;
        NonCopyable(const NonCopyable&)  = delete;
        NonCopyable(const NonCopyable&&) = delete;
        NonCopyable& operator=(const NonCopyable&) = delete;
        NonCopyable& operator=(const NonCopyable&&) = delete;
    };

} // namespace TRITON_SERVER