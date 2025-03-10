// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 1998 - 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#ifndef dealii_exceptions_h
#define dealii_exceptions_h

#include <deal.II/base/config.h>

#include <deal.II/base/numbers.h>

DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#include <Kokkos_Macros.hpp>
#if KOKKOS_VERSION >= 40200
#  include <Kokkos_Abort.hpp>
#else
#  include <Kokkos_Core.hpp>
#endif
DEAL_II_ENABLE_EXTRA_DIAGNOSTICS

#include <exception>
#include <ostream>
#include <string>
#include <type_traits>

DEAL_II_NAMESPACE_OPEN


/**
 * This class is the base class for all exception classes. Do not use its
 * methods and variables directly since the interface and mechanism may be
 * subject to change. Rather create new exception classes using the
 * <tt>DeclException</tt> macro family.
 *
 * See the
 * @ref Exceptions
 * topic for more details on this class and what can be done with classes
 * derived from it.
 *
 * @ingroup Exceptions
 */
class ExceptionBase : public std::exception
{
public:
  /**
   * Default constructor.
   */
  ExceptionBase();

  /**
   * Copy constructor.
   */
  ExceptionBase(const ExceptionBase &exc);

  /**
   * Destructor.
   */
  virtual ~ExceptionBase() noexcept override = default;

  /**
   * Copy operator. This operator is deleted since exception objects
   * are not copyable.
   */
  ExceptionBase
  operator=(const ExceptionBase &) = delete;

  /**
   * Set the file name and line of where the exception appeared as well as the
   * violated condition and the name of the exception as a char pointer. This
   * function also populates the stacktrace.
   */
  void
  set_fields(const char *file,
             const int   line,
             const char *function,
             const char *cond,
             const char *exc_name);


  /**
   * Override the standard function that returns the description of the error.
   */
  virtual const char *
  what() const noexcept override;

  /**
   * Get exception name.
   */
  const char *
  get_exc_name() const;

  /**
   * Print out the general part of the error information.
   */
  void
  print_exc_data(std::ostream &out) const;

  /**
   * Print more specific information about the exception which occurred.
   * Overload this function in your own exception classes.
   */
  virtual void
  print_info(std::ostream &out) const;

  /**
   * Print a stacktrace, if one has been recorded previously, to the given
   * stream.
   */
  void
  print_stack_trace(std::ostream &out) const;

protected:
  /**
   * Name of the file this exception happens in.
   */
  const char *file;

  /**
   * Line number in this file.
   */
  unsigned int line;

  /**
   * Name of the function, pretty printed.
   */
  const char *function;

  /**
   * The violated condition, as a string.
   */
  const char *cond;

  /**
   * Name of the exception and call sequence.
   */
  const char *exc;

  /**
   * The number of stacktrace frames that are stored in the following variable.
   * Zero if the system does not support stack traces.
   */
  int n_stacktrace_frames;

#ifdef DEAL_II_HAVE_GLIBC_STACKTRACE
  /**
   * Array of pointers that contains the raw stack trace.
   */
  void *raw_stacktrace[25];
#endif

private:
  /**
   * Internal function that generates the c_string. Called by what().
   */
  void
  generate_message() const;

  /**
   * A pointer to the c_string that will be printed by what(). It is populated
   * by generate_message()
   */
  mutable std::string what_str;
};

#ifndef DOXYGEN

/**
 * Declare an exception class derived from ExceptionBase without parameters.
 *
 * @note This and similar macro names are examples of preprocessor definitions
 * in the deal.II library that are not prefixed by a string that likely makes
 * them unique to deal.II. As a consequence, it is possible that other
 * libraries your code interfaces with define the same name, and the result
 * will be name collisions (see
 * https://en.wikipedia.org/wiki/Name_collision). One can <code>\#undef</code>
 * this macro, as well as all other macros defined by deal.II that are not
 * prefixed with either <code>DEAL</code> or <code>deal</code>, by including
 * the header <code>deal.II/base/undefine_macros.h</code> after all other
 * deal.II headers have been included.
 *
 * @ingroup Exceptions
 */
#  define DeclException0(Exception0)                \
    class Exception0 : public dealii::ExceptionBase \
    {}


/**
 * Declare an exception class derived from ExceptionBase that can take one
 * runtime argument, but if none is given in the place where you want to throw
 * the exception, it simply reverts to the default text provided when
 * declaring the exception class through this macro.
 *
 * @note This and similar macro names are examples of preprocessor definitions
 * in the deal.II library that are not prefixed by a string that likely makes
 * them unique to deal.II. As a consequence, it is possible that other
 * libraries your code interfaces with define the same name, and the result
 * will be name collisions (see
 * https://en.wikipedia.org/wiki/Name_collision). One can <code>\#undef</code>
 * this macro, as well as all other macros defined by deal.II that are not
 * prefixed with either <code>DEAL</code> or <code>deal</code>, by including
 * the header <code>deal.II/base/undefine_macros.h</code> after all other
 * deal.II headers have been included.
 *
 * @ingroup Exceptions
 */
#  define DeclExceptionMsg(Exception, defaulttext)    \
    class Exception : public dealii::ExceptionBase    \
    {                                                 \
    public:                                           \
      Exception(const std::string &msg = defaulttext) \
        : arg(msg)                                    \
      {}                                              \
      virtual ~Exception() noexcept                   \
      {}                                              \
      virtual void                                    \
      print_info(std::ostream &out) const override    \
      {                                               \
        out << "    " << arg << std::endl;            \
      }                                               \
                                                      \
    private:                                          \
      const std::string arg;                          \
    }

/**
 * Declare an exception class derived from ExceptionBase with one additional
 * parameter.
 *
 * @note This and similar macro names are examples of preprocessor definitions
 * in the deal.II library that are not prefixed by a string that likely makes
 * them unique to deal.II. As a consequence, it is possible that other
 * libraries your code interfaces with define the same name, and the result
 * will be name collisions (see
 * https://en.wikipedia.org/wiki/Name_collision). One can <code>\#undef</code>
 * this macro, as well as all other macros defined by deal.II that are not
 * prefixed with either <code>DEAL</code> or <code>deal</code>, by including
 * the header <code>deal.II/base/undefine_macros.h</code> after all other
 * deal.II headers have been included.
 *
 * @ingroup Exceptions
 */
#  define DeclException1(Exception1, type1, outsequence) \
    class Exception1 : public dealii::ExceptionBase      \
    {                                                    \
    public:                                              \
      Exception1(type1 const &a1)                        \
        : arg1(a1)                                       \
      {}                                                 \
      virtual ~Exception1() noexcept                     \
      {}                                                 \
      virtual void                                       \
      print_info(std::ostream &out) const override       \
      {                                                  \
        out << "    " outsequence << std::endl;          \
      }                                                  \
                                                         \
    private:                                             \
      type1 const arg1;                                  \
    }


/**
 * Declare an exception class derived from ExceptionBase with two additional
 * parameters.
 *
 * @note This and similar macro names are examples of preprocessor definitions
 * in the deal.II library that are not prefixed by a string that likely makes
 * them unique to deal.II. As a consequence, it is possible that other
 * libraries your code interfaces with define the same name, and the result
 * will be name collisions (see
 * https://en.wikipedia.org/wiki/Name_collision). One can <code>\#undef</code>
 * this macro, as well as all other macros defined by deal.II that are not
 * prefixed with either <code>DEAL</code> or <code>deal</code>, by including
 * the header <code>deal.II/base/undefine_macros.h</code> after all other
 * deal.II headers have been included.
 *
 * @ingroup Exceptions
 */
#  define DeclException2(Exception2, type1, type2, outsequence) \
    class Exception2 : public dealii::ExceptionBase             \
    {                                                           \
    public:                                                     \
      Exception2(type1 const &a1, type2 const &a2)              \
        : arg1(a1)                                              \
        , arg2(a2)                                              \
      {}                                                        \
      virtual ~Exception2() noexcept                            \
      {}                                                        \
      virtual void                                              \
      print_info(std::ostream &out) const override              \
      {                                                         \
        out << "    " outsequence << std::endl;                 \
      }                                                         \
                                                                \
    private:                                                    \
      type1 const arg1;                                         \
      type2 const arg2;                                         \
    }


/**
 * Declare an exception class derived from ExceptionBase with three additional
 * parameters.
 *
 * @note This and similar macro names are examples of preprocessor definitions
 * in the deal.II library that are not prefixed by a string that likely makes
 * them unique to deal.II. As a consequence, it is possible that other
 * libraries your code interfaces with define the same name, and the result
 * will be name collisions (see
 * https://en.wikipedia.org/wiki/Name_collision). One can <code>\#undef</code>
 * this macro, as well as all other macros defined by deal.II that are not
 * prefixed with either <code>DEAL</code> or <code>deal</code>, by including
 * the header <code>deal.II/base/undefine_macros.h</code> after all other
 * deal.II headers have been included.
 *
 * @ingroup Exceptions
 */
#  define DeclException3(Exception3, type1, type2, type3, outsequence) \
    class Exception3 : public dealii::ExceptionBase                    \
    {                                                                  \
    public:                                                            \
      Exception3(type1 const &a1, type2 const &a2, type3 const &a3)    \
        : arg1(a1)                                                     \
        , arg2(a2)                                                     \
        , arg3(a3)                                                     \
      {}                                                               \
      virtual ~Exception3() noexcept                                   \
      {}                                                               \
      virtual void                                                     \
      print_info(std::ostream &out) const override                     \
      {                                                                \
        out << "    " outsequence << std::endl;                        \
      }                                                                \
                                                                       \
    private:                                                           \
      type1 const arg1;                                                \
      type2 const arg2;                                                \
      type3 const arg3;                                                \
    }


/**
 * Declare an exception class derived from ExceptionBase with four additional
 * parameters.
 *
 * @note This and similar macro names are examples of preprocessor definitions
 * in the deal.II library that are not prefixed by a string that likely makes
 * them unique to deal.II. As a consequence, it is possible that other
 * libraries your code interfaces with define the same name, and the result
 * will be name collisions (see
 * https://en.wikipedia.org/wiki/Name_collision). One can <code>\#undef</code>
 * this macro, as well as all other macros defined by deal.II that are not
 * prefixed with either <code>DEAL</code> or <code>deal</code>, by including
 * the header <code>deal.II/base/undefine_macros.h</code> after all other
 * deal.II headers have been included.
 *
 * @ingroup Exceptions
 */
#  define DeclException4(Exception4, type1, type2, type3, type4, outsequence) \
    class Exception4 : public dealii::ExceptionBase                           \
    {                                                                         \
    public:                                                                   \
      Exception4(type1 const &a1,                                             \
                 type2 const &a2,                                             \
                 type3 const &a3,                                             \
                 type4 const &a4)                                             \
        : arg1(a1)                                                            \
        , arg2(a2)                                                            \
        , arg3(a3)                                                            \
        , arg4(a4)                                                            \
      {}                                                                      \
      virtual ~Exception4() noexcept                                          \
      {}                                                                      \
      virtual void                                                            \
      print_info(std::ostream &out) const override                            \
      {                                                                       \
        out << "    " outsequence << std::endl;                               \
      }                                                                       \
                                                                              \
    private:                                                                  \
      type1 const arg1;                                                       \
      type2 const arg2;                                                       \
      type3 const arg3;                                                       \
      type4 const arg4;                                                       \
    }


/**
 * Declare an exception class derived from ExceptionBase with five additional
 * parameters.
 *
 * @note This and similar macro names are examples of preprocessor definitions
 * in the deal.II library that are not prefixed by a string that likely makes
 * them unique to deal.II. As a consequence, it is possible that other
 * libraries your code interfaces with define the same name, and the result
 * will be name collisions (see
 * https://en.wikipedia.org/wiki/Name_collision). One can <code>\#undef</code>
 * this macro, as well as all other macros defined by deal.II that are not
 * prefixed with either <code>DEAL</code> or <code>deal</code>, by including
 * the header <code>deal.II/base/undefine_macros.h</code> after all other
 * deal.II headers have been included.
 *
 * @ingroup Exceptions
 */
#  define DeclException5(                                       \
    Exception5, type1, type2, type3, type4, type5, outsequence) \
    class Exception5 : public dealii::ExceptionBase             \
    {                                                           \
    public:                                                     \
      Exception5(type1 const &a1,                               \
                 type2 const &a2,                               \
                 type3 const &a3,                               \
                 type4 const &a4,                               \
                 type5 const &a5)                               \
        : arg1(a1)                                              \
        , arg2(a2)                                              \
        , arg3(a3)                                              \
        , arg4(a4)                                              \
        , arg5(a5)                                              \
      {}                                                        \
      virtual ~Exception5() noexcept                            \
      {}                                                        \
      virtual void                                              \
      print_info(std::ostream &out) const override              \
      {                                                         \
        out << "    " outsequence << std::endl;                 \
      }                                                         \
                                                                \
    private:                                                    \
      type1 const arg1;                                         \
      type2 const arg2;                                         \
      type3 const arg3;                                         \
      type4 const arg4;                                         \
      type5 const arg5;                                         \
    }

#else /*ifndef DOXYGEN*/

// Dummy definitions for doxygen:

/**
 * Declare an exception class derived from ExceptionBase without parameters.
 *
 * @note This and similar macro names are examples of preprocessor definitions
 * in the deal.II library that are not prefixed by a string that likely makes
 * them unique to deal.II. As a consequence, it is possible that other
 * libraries your code interfaces with define the same name, and the result
 * will be name collisions (see
 * https://en.wikipedia.org/wiki/Name_collision). One can <code>\#undef</code>
 * this macro, as well as all other macros defined by deal.II that are not
 * prefixed with either <code>DEAL</code> or <code>deal</code>, by including
 * the header <code>deal.II/base/undefine_macros.h</code> after all other
 * deal.II headers have been included.
 *
 * @ingroup Exceptions
 */
#  define DeclException0(Exception0) \
    /** @ingroup Exceptions */       \
    static dealii::ExceptionBase &Exception0()

/**
 * Declare an exception class derived from ExceptionBase that can take one
 * runtime argument, but if none is given in the place where you want to throw
 * the exception, it simply reverts to the default text provided when
 * declaring the exception class through this macro.
 *
 * @note This and similar macro names are examples of preprocessor definitions
 * in the deal.II library that are not prefixed by a string that likely makes
 * them unique to deal.II. As a consequence, it is possible that other
 * libraries your code interfaces with define the same name, and the result
 * will be name collisions (see
 * https://en.wikipedia.org/wiki/Name_collision). One can <code>\#undef</code>
 * this macro, as well as all other macros defined by deal.II that are not
 * prefixed with either <code>DEAL</code> or <code>deal</code>, by including
 * the header <code>deal.II/base/undefine_macros.h</code> after all other
 * deal.II headers have been included.
 *
 * @ingroup Exceptions
 */
#  define DeclExceptionMsg(Exception, defaulttext) \
    /** @ingroup Exceptions */                     \
    /** @dealiiExceptionMessage{defaulttext} */    \
    static dealii::ExceptionBase &Exception()

/**
 * Declare an exception class derived from ExceptionBase with one additional
 * parameter.
 *
 * @note This and similar macro names are examples of preprocessor definitions
 * in the deal.II library that are not prefixed by a string that likely makes
 * them unique to deal.II. As a consequence, it is possible that other
 * libraries your code interfaces with define the same name, and the result
 * will be name collisions (see
 * https://en.wikipedia.org/wiki/Name_collision). One can <code>\#undef</code>
 * this macro, as well as all other macros defined by deal.II that are not
 * prefixed with either <code>DEAL</code> or <code>deal</code>, by including
 * the header <code>deal.II/base/undefine_macros.h</code> after all other
 * deal.II headers have been included.
 *
 * @ingroup Exceptions
 */
#  define DeclException1(Exception1, type1, outsequence) \
    /** @ingroup Exceptions */                           \
    /** @dealiiExceptionMessage{outsequence} */          \
    static dealii::ExceptionBase &Exception1(type1 arg1)


/**
 * Declare an exception class derived from ExceptionBase with two additional
 * parameters.
 *
 * @note This and similar macro names are examples of preprocessor definitions
 * in the deal.II library that are not prefixed by a string that likely makes
 * them unique to deal.II. As a consequence, it is possible that other
 * libraries your code interfaces with define the same name, and the result
 * will be name collisions (see
 * https://en.wikipedia.org/wiki/Name_collision). One can <code>\#undef</code>
 * this macro, as well as all other macros defined by deal.II that are not
 * prefixed with either <code>DEAL</code> or <code>deal</code>, by including
 * the header <code>deal.II/base/undefine_macros.h</code> after all other
 * deal.II headers have been included.
 *
 * @ingroup Exceptions
 */
#  define DeclException2(Exception2, type1, type2, outsequence) \
    /** @ingroup Exceptions */                                  \
    /** @dealiiExceptionMessage{outsequence} */                 \
    static dealii::ExceptionBase &Exception2(type1 arg1, type2 arg2)


/**
 * Declare an exception class derived from ExceptionBase with three additional
 * parameters.
 *
 * @note This and similar macro names are examples of preprocessor definitions
 * in the deal.II library that are not prefixed by a string that likely makes
 * them unique to deal.II. As a consequence, it is possible that other
 * libraries your code interfaces with define the same name, and the result
 * will be name collisions (see
 * https://en.wikipedia.org/wiki/Name_collision). One can <code>\#undef</code>
 * this macro, as well as all other macros defined by deal.II that are not
 * prefixed with either <code>DEAL</code> or <code>deal</code>, by including
 * the header <code>deal.II/base/undefine_macros.h</code> after all other
 * deal.II headers have been included.
 *
 * @ingroup Exceptions
 */
#  define DeclException3(Exception3, type1, type2, type3, outsequence) \
    /** @ingroup Exceptions */                                         \
    /** @dealiiExceptionMessage{outsequence} */                        \
    static dealii::ExceptionBase &Exception3(type1 arg1, type2 arg2, type3 arg3)


/**
 * Declare an exception class derived from ExceptionBase with four additional
 * parameters.
 *
 * @note This and similar macro names are examples of preprocessor definitions
 * in the deal.II library that are not prefixed by a string that likely makes
 * them unique to deal.II. As a consequence, it is possible that other
 * libraries your code interfaces with define the same name, and the result
 * will be name collisions (see
 * https://en.wikipedia.org/wiki/Name_collision). One can <code>\#undef</code>
 * this macro, as well as all other macros defined by deal.II that are not
 * prefixed with either <code>DEAL</code> or <code>deal</code>, by including
 * the header <code>deal.II/base/undefine_macros.h</code> after all other
 * deal.II headers have been included.
 *
 * @ingroup Exceptions
 */
#  define DeclException4(Exception4, type1, type2, type3, type4, outsequence) \
    /** @ingroup Exceptions */                                                \
    /** @dealiiExceptionMessage{outsequence} */                               \
    static dealii::ExceptionBase &Exception4(type1 arg1,                      \
                                             type2 arg2,                      \
                                             type3 arg3,                      \
                                             type4 arg4)


/**
 * Declare an exception class derived from ExceptionBase with five additional
 * parameters.
 *
 * @note This and similar macro names are examples of preprocessor definitions
 * in the deal.II library that are not prefixed by a string that likely makes
 * them unique to deal.II. As a consequence, it is possible that other
 * libraries your code interfaces with define the same name, and the result
 * will be name collisions (see
 * https://en.wikipedia.org/wiki/Name_collision). One can <code>\#undef</code>
 * this macro, as well as all other macros defined by deal.II that are not
 * prefixed with either <code>DEAL</code> or <code>deal</code>, by including
 * the header <code>deal.II/base/undefine_macros.h</code> after all other
 * deal.II headers have been included.
 *
 * @ingroup Exceptions
 */
#  define DeclException5(                                       \
    Exception5, type1, type2, type3, type4, type5, outsequence) \
    /** @ingroup Exceptions */                                  \
    /** @dealiiExceptionMessage{outsequence} */                 \
    static dealii::ExceptionBase &Exception5(                   \
      type1 arg1, type2 arg2, type3 arg3, type4 arg4, type5 arg5)

#endif /*ifndef DOXYGEN*/


/**
 * Declare some exceptions that occur over and over. This way, you can simply
 * use these exceptions, instead of having to declare them locally in your
 * class. The namespace in which these exceptions are declared is later
 * included into the global namespace by
 * @code
 * using namespace StandardExceptions;
 * @endcode
 *
 * @ingroup Exceptions
 */
namespace StandardExceptions
{
  /**
   * @addtogroup Exceptions
   * @{
   */

  /**
   * Exception denoting a division by zero.
   */
  DeclExceptionMsg(ExcDivideByZero,
                   "A piece of code is attempting a division by zero. This is "
                   "likely going to lead to results that make no sense.");

  /**
   * Exception raised if a number is not finite.
   *
   * This exception should be used to catch infinite or not a number results
   * of arithmetic operations that do not result from a division by zero (use
   * ExcDivideByZero for those).
   *
   * The exception uses std::complex as its argument to ensure that we can use
   * it for all scalar arguments (real or complex-valued).
   */
  DeclException1(
    ExcNumberNotFinite,
    std::complex<double>,
    << "In a significant number of places, deal.II checks that some intermediate "
    << "value is a finite number (as opposed to plus or minus infinity, or "
    << "NaN/Not a Number). In the current function, we encountered a number "
    << "that is not finite (its value is " << arg1 << " and therefore "
    << "violates the current assertion).\n\n"
    << "This may be due to the fact that some operation in this function "
    << "created such a value, or because one of the arguments you passed "
    << "to the function already had this value from some previous "
    << "operation. In the latter case, this function only triggered the "
    << "error but may not actually be responsible for the computation of "
    << "the number that is not finite.\n\n"
    << "There are two common cases where this situation happens. First, your "
    << "code (or something in deal.II) divides by zero in a place where this "
    << "should not happen. Or, you are trying to solve a linear system "
    << "with an unsuitable solver (such as an indefinite or non-symmetric "
    << "linear system using a Conjugate Gradient solver); such attempts "
    << "oftentimes yield an operation somewhere that tries to divide "
    << "by zero or take the square root of a negative value.\n\n"
    << "In any case, when trying to find the source of the error, "
    << "recall that the location where you are getting this error is "
    << "simply the first place in the program where there is a check "
    << "that a number (e.g., an element of a solution vector) is in fact "
    << "finite, but that the actual error that computed the number "
    << "may have happened far earlier. To find this location, you "
    << "may want to add checks for finiteness in places of your "
    << "program visited before the place where this error is produced. "
    << "One way to check for finiteness is to use the 'AssertIsFinite' "
    << "macro.");

  /**
   * Trying to allocate a new object failed due to lack of free memory.
   */
  DeclException1(ExcOutOfMemory,
                 std::size_t,
                 "Your program tried to allocate some memory but this "
                 "allocation failed. Typically, this either means that "
                 "you simply do not have enough memory in your system, "
                 "or that you are (erroneously) trying to allocate "
                 "a chunk of memory that is simply beyond all reasonable "
                 "size, for example because the size of the object has "
                 "been computed incorrectly."
                 "\n\n"
                 "In the current case, the request was for "
                   << arg1 << " bytes.");

  /**
   * A memory handler reached a point where all allocated objects should have
   * been released. Since this exception is thrown, some were still allocated.
   */
  DeclException1(ExcMemoryLeak,
                 int,
                 << "Destroying memory handler while " << arg1
                 << " objects are still allocated.");

  /**
   * An error occurred reading or writing a file.
   */
  DeclExceptionMsg(ExcIO,
                   "An input/output error has occurred. There are a number of "
                   "reasons why this may be happening, both for reading and "
                   "writing operations."
                   "\n\n"
                   "If this happens during an operation that tries to read "
                   "data: First, you may be "
                   "trying to read from a file that doesn't exist or that is "
                   "not readable given its file permissions. Second, deal.II "
                   "uses this error at times if it tries to "
                   "read information from a file but where the information "
                   "in the file does not correspond to the expected format. "
                   "An example would be a truncated file, or a mesh file "
                   "that contains not only sections that describe the "
                   "vertices and cells, but also sections for additional "
                   "data that deal.II does not understand."
                   "\n\n"
                   "If this happens during an operation that tries to write "
                   "data: you may be trying to write to a file to which file "
                   "or directory permissions do not allow you to write. A "
                   "typical example is where you specify an output file in "
                   "a directory that does not exist.");

  /**
   * An error occurred opening the named file.
   *
   * The constructor takes a single argument of type <tt>std::string</tt> naming
   * the file.
   */
  DeclException1(ExcFileNotOpen,
                 std::string,
                 << "Could not open file " << arg1
                 << "."
                    "\n\n"
                    "If this happens during an operation that tries to read "
                    "data: you may be "
                    "trying to read from a file that doesn't exist or that is "
                    "not readable given its file permissions."
                    "\n\n"
                    "If this happens during an operation that tries to write "
                    "data: you may be trying to write to a file to which file "
                    "or directory permissions do not allow you to write. A "
                    "typical example is where you specify an output file in "
                    "a directory that does not exist.");

  /**
   * Exception denoting a part of the library or application program that has
   * not yet been implemented. In many cases, this only indicates that there
   * wasn't much need for something yet, not that this is difficult to
   * implement. It is therefore quite worth the effort to take a look at the
   * corresponding place and see whether it can be implemented without too
   * much effort.
   */
  DeclExceptionMsg(ExcNotImplemented,
                   "You are trying to use functionality in deal.II that is "
                   "currently not implemented. In many cases, this indicates "
                   "that there simply didn't appear much of a need for it, or "
                   "that the author of the original code did not have the "
                   "time to implement a particular case. If you hit this "
                   "exception, it is therefore worth the time to look into "
                   "the code to find out whether you may be able to "
                   "implement the missing functionality. If you do, please "
                   "consider providing a patch to the deal.II development "
                   "sources (see the deal.II website on how to contribute).");

  /**
   * This exception usually indicates that some condition which the programmer
   * thinks must be satisfied at a certain point in an algorithm, is not
   * fulfilled. This might be due to some programming error above, due to
   * changes to the algorithm that did not preserve this assertion, or due to
   * assumptions the programmer made that are not valid at all (i.e. the
   * exception is thrown although there is no error here). Within the library,
   * this exception is most often used when we write some kind of complicated
   * algorithm and are not yet sure whether we got it right; we then put in
   * assertions after each part of the algorithm that check for some
   * conditions that should hold there, and throw an exception if they do not.
   *
   * We usually leave in these assertions even after we are confident that the
   * implementation is correct, since if someone later changes or extends the
   * algorithm, these exceptions will indicate to them if they violate
   * assumptions that are used later in the algorithm. Furthermore, it
   * sometimes happens that an algorithm does not work in very rare corner
   * cases. These cases will then be trapped sooner or later by the exception,
   * so that the algorithm can then be fixed for these cases as well.
   */
  DeclExceptionMsg(ExcInternalError,
                   "This exception -- which is used in many places in the "
                   "library -- usually indicates that some condition which "
                   "the author of the code thought must be satisfied at a "
                   "certain point in an algorithm, is not fulfilled. An "
                   "example would be that the first part of an algorithm "
                   "sorts elements of an array in ascending order, and "
                   "a second part of the algorithm later encounters an "
                   "element that is not larger than the previous one."
                   "\n\n"
                   "There is usually not very much you can do if you "
                   "encounter such an exception since it indicates an error "
                   "in deal.II, not in your own program. Try to come up with "
                   "the smallest possible program that still demonstrates "
                   "the error and contact the deal.II mailing lists with it "
                   "to obtain help.");

  /**
   * This exception is used in functions that may not be called (i.e. in pure
   * functions) but could not be declared pure since the class is intended to
   * be used anyway, even though the respective function may only be called if
   * a derived class is used.
   */
  DeclExceptionMsg(
    ExcPureFunctionCalled,
    "You (or a place in the library) are trying to call a "
    "function that is declared as a virtual function in a "
    "base class but that has not been overridden in your "
    "derived class."
    "\n\n"
    "This exception happens in cases where the base class "
    "cannot provide a useful default implementation for "
    "the virtual function, but where we also do not want "
    "to mark the function as abstract (i.e., with '=0' at the end) "
    "because the function is not essential to the class in many "
    "contexts. In cases like this, the base class provides "
    "a dummy implementation that makes the compiler happy, but "
    "that then throws the current exception."
    "\n\n"
    "A concrete example would be the 'Function' class. It declares "
    "the existence of 'value()' and 'gradient()' member functions, "
    "and both are marked as 'virtual'. Derived classes have to "
    "override these functions for the values and gradients of a "
    "particular function. On the other hand, not every function "
    "has a gradient, and even for those that do, not every program "
    "actually needs to evaluate it. Consequently, there is no "
    "*requirement* that a derived class actually override the "
    "'gradient()' function (as there would be had it been marked "
    "as abstract). But, since the base class cannot know how to "
    "compute the gradient, if a derived class does not override "
    "the 'gradient()' function and it is called anyway, then the "
    "default implementation in the base class will simply throw "
    "an exception."
    "\n\n"
    "The exception you see is what happens in cases such as the "
    "one just illustrated. To fix the problem, you need to "
    "investigate whether the function being called should indeed have "
    "been called; if the answer is 'yes', then you need to "
    "implement the missing override in your class.");

  /**
   * This exception is used if some user function is not provided.
   */
  DeclException1(ExcFunctionNotProvided,
                 std::string,
                 << "Please provide an implementation for the function \""
                 << arg1 << "\"");

  /**
   * This exception is used if some user function returns nonzero exit codes.
   */
  DeclException2(
    ExcFunctionNonzeroReturn,
    std::string,
    int,
    << "The function \"" << arg1 << "\" returned the nonzero value " << arg2
    << ", but the calling site expected the return value to be zero. "
       "This error often happens when the function in question is a 'callback', "
       "that is a user-provided function called from somewhere within deal.II "
       "or within an external library such as PETSc, Trilinos, SUNDIALS, etc., "
       "that expect these callbacks to indicate errors via nonzero return "
       "codes.");

  /**
   * This exception is used if some object is found uninitialized.
   */
  DeclException0(ExcNotInitialized);

  /**
   * The object is in a state not suitable for this operation.
   */
  DeclException0(ExcInvalidState);

  /**
   * This exception is raised if a functionality is not possible in the given
   * dimension. Mostly used to throw function calls in 1d.
   *
   * The constructor takes a single <tt>int</tt>, denoting the dimension.
   */
  DeclException1(ExcImpossibleInDim,
                 int,
                 << "You are trying to execute functionality that is "
                 << "impossible in " << arg1
                 << "d or simply does not make any sense.");

  /**
   * This exception is raised if a functionality is not possible in the given
   * combination of dimension and space-dimension.
   *
   * The constructor takes two <tt>int</tt>, denoting the dimension and the
   * space dimension.
   */
  DeclException2(ExcImpossibleInDimSpacedim,
                 int,
                 int,
                 << "You are trying to execute functionality that is "
                 << "impossible in dimensions <" << arg1 << ',' << arg2
                 << "> or simply does not make any sense.");


  /**
   * A number is zero, but it should not be here.
   */
  DeclExceptionMsg(ExcZero,
                   "In a check in the code, deal.II encountered a zero in "
                   "a place where this does not make sense. See the condition "
                   "that was being checked and that is printed further up "
                   "in the error message to get more information on what "
                   "the erroneous zero corresponds to.");

  /**
   * The object should have been filled with something before this member
   * function is called.
   */
  DeclExceptionMsg(ExcEmptyObject,
                   "The object you are trying to access is empty but it makes "
                   "no sense to attempt the operation you are trying on an "
                   "empty object.");

  /**
   * This exception is raised whenever the sizes of two objects were assumed
   * to be equal, but were not.
   *
   * Parameters to the constructor are the first and second size, both of type
   * <tt>int</tt>.
   */
  DeclException2(ExcDimensionMismatch,
                 std::size_t,
                 std::size_t,
                 << "Two sizes or dimensions were supposed to be equal, "
                 << "but aren't. They are " << arg1 << " and " << arg2 << '.');

  /**
   * This exception is raised whenever deal.II cannot convert between integer
   * types.
   */
  DeclException2(
    ExcInvalidIntegerConversion,
    long long,
    long long,
    << "Two integers should be equal to each other after a type conversion but "
    << "aren't. A typical cause of this problem is that the integral types "
    << "used by deal.II and an external library are different (e.g., one uses "
    << "32-bit integers and the other uses 64-bit integers). The integers are "
    << arg1 << " and " << arg2 << '.');

  /**
   * The first dimension should be either equal to the second or the third,
   * but it is neither.
   */
  DeclException3(ExcDimensionMismatch2,
                 std::size_t,
                 std::size_t,
                 std::size_t,
                 << "The size or dimension of one object, " << arg1
                 << " was supposed to be "
                 << "equal to one of two values, but isn't. The two possible "
                 << "values are " << arg2 << " and " << arg3 << '.');

  /**
   * This exception indicates that an index is not within the expected range.
   * For example, it may be that you are trying to access an element of a
   * vector which does not exist.
   *
   * The constructor takes three <tt>std::size_t</tt> arguments, namely
   * <ol>
   * <li> the violating index
   * <li> the lower bound
   * <li> the upper bound plus one
   * </ol>
   */
  DeclException3(
    ExcIndexRange,
    std::size_t,
    std::size_t,
    std::size_t,
    << "Index " << arg1 << " is not in the half-open range [" << arg2 << ','
    << arg3 << ")."
    << (arg2 == arg3 ?
          " In the current case, this half-open range is in fact empty, "
          "suggesting that you are accessing an element of an empty "
          "collection such as a vector that has not been set to the "
          "correct size." :
          ""));

  /**
   * This exception indicates that an index is not within the expected range.
   * For example, it may be that you are trying to access an element of a
   * vector which does not exist.
   *
   * The constructor takes three arguments, namely
   * <ol>
   * <li> the violating index
   * <li> the lower bound
   * <li> the upper bound plus one
   * </ol>
   *
   * This generic exception differs from ExcIndexRange by allowing to specify
   * the type of indices.
   */
  template <typename T>
  DeclException3(
    ExcIndexRangeType,
    T,
    T,
    T,
    << "Index " << arg1 << " is not in the half-open range [" << arg2 << ','
    << arg3 << ")."
    << (arg2 == arg3 ?
          " In the current case, this half-open range is in fact empty, "
          "suggesting that you are accessing an element of an empty "
          "collection such as a vector that has not been set to the "
          "correct size." :
          ""));

  /**
   * A number is too small.
   */
  DeclException2(ExcLowerRange,
                 int,
                 int,
                 << "Number " << arg1 << " must be larger than or equal "
                 << arg2 << '.');

  /**
   * A generic exception definition for the ExcLowerRange above.
   */
  template <typename T>
  DeclException2(ExcLowerRangeType,
                 T,
                 T,
                 << "Number " << arg1 << " must be larger than or equal "
                 << arg2 << '.');

  /**
   * This exception indicates that the first argument should be an integer
   * multiple of the second, but is not.
   */
  DeclException2(ExcNotMultiple,
                 int,
                 int,
                 << "Division " << arg1 << " by " << arg2
                 << " has remainder different from zero.");

  /**
   * This exception is thrown if the iterator you access has corrupted data.
   * It might for instance be, that the container it refers does not have an
   * entry at the point the iterator refers.
   *
   * Typically, this will be an internal error of deal.II, because the
   * increment and decrement operators should never yield an invalid iterator.
   */
  DeclExceptionMsg(ExcInvalidIterator,
                   "You are trying to use an iterator, but the iterator is "
                   "in an invalid state. This may indicate that the iterator "
                   "object has not been initialized, or that it has been "
                   "moved beyond the end of the range of valid elements.");

  /**
   * This exception is thrown if the iterator you incremented or decremented
   * was already at its final state.
   */
  DeclExceptionMsg(ExcIteratorPastEnd,
                   "You are trying to use an iterator, but the iterator is "
                   "pointing past the end of the range of valid elements. "
                   "It is not valid to dereference the iterator in this "
                   "case.");

  /**
   * This exception works around a design flaw in the <tt>DeclException0</tt>
   * macro: exceptions declared through DeclException0 do not allow one to
   * specify a message that is displayed when the exception is raised, as
   * opposed to the other exceptions which allow to show a text along with the
   * given parameters.
   *
   * When throwing this exception, you can give a message as a
   * <tt>std::string</tt> as argument to the exception that is then displayed.
   * The argument can, of course, be constructed at run-time, for example
   * including the name of a file that can't be opened, or any other text you
   * may want to assemble from different pieces.
   */
  DeclException1(ExcMessage, std::string, << arg1);

  /**
   * Parallel vectors with ghost elements are read-only vectors.
   */
  DeclExceptionMsg(ExcGhostsPresent,
                   "You are trying an operation on a vector that is only "
                   "allowed if the vector has no ghost elements, but the "
                   "vector you are operating on does have ghost elements."
                   "\n\n"
                   "Specifically, there are two kinds of operations that "
                   "are typically not allowed on vectors with ghost elements. "
                   "First, vectors with ghost elements are read-only "
                   "and cannot appear in operations that write into these "
                   "vectors. Second, reduction operations (such as computing "
                   "the norm of a vector, or taking dot products between "
                   "vectors) are not allowed to ensure that each vector "
                   "element is counted only once (as opposed to once for "
                   "the owner of the element plus once for each process "
                   "on which the element is stored as a ghost copy)."
                   "\n\n"
                   "See the glossary entry on 'Ghosted vectors' for more "
                   "information.");

  /**
   * Exception indicating that one of the cells in the input to
   * Triangulation::create_triangulation() or a related function cannot be used.
   */
  DeclException1(ExcGridHasInvalidCell,
                 int,
                 << "Something went wrong when making cell " << arg1
                 << ". Read the docs and the source code "
                 << "for more information.");

  /**
   * Some of our numerical classes allow for setting all entries to zero using
   * the assignment operator <tt>=</tt>.
   *
   * In many cases, this assignment operator makes sense <b>only</b> for the
   * argument zero. In other cases, this exception is thrown.
   */
  DeclExceptionMsg(ExcScalarAssignmentOnlyForZeroValue,
                   "You are trying an operation of the form 'vector = C', "
                   "'matrix = C', or 'tensor = C' with a nonzero scalar value "
                   "'C'. However, such assignments are only allowed if the "
                   "C is zero, since the semantics for assigning any other "
                   "value are not clear. For example: one could interpret "
                   "assigning a matrix a value of 1 to mean the matrix has a "
                   "norm of 1, the matrix is the identity matrix, or the "
                   "matrix contains only 1s. Similar problems exist with "
                   "vectors and tensors. Hence, to avoid this ambiguity, such "
                   "assignments are not permitted.");

  /**
   * This function requires support for the LAPACK library.
   */
  DeclExceptionMsg(
    ExcNeedsLAPACK,
    "You are attempting to use functionality that is only available "
    "if deal.II was configured to use LAPACK, but when you configured "
    "the library, cmake did not find a valid LAPACK library."
    "\n\n"
    "You will have to ensure that your system has a usable LAPACK "
    "installation and re-install deal.II, making sure that cmake "
    "finds the LAPACK installation. You can check this by "
    "looking at the summary printed at the end of the cmake "
    "output.");

  /**
   * This function requires support for the HDF5 library.
   */
  DeclExceptionMsg(
    ExcNeedsHDF5,
    "You are attempting to use functionality that requires that deal.II is configured "
    "with HDF5 support. However, when you called 'cmake', HDF5 support "
    "was not detected."
    "\n\n"
    "You will have to ensure that your system has a usable HDF5 "
    "installation and re-install deal.II, making sure that cmake "
    "finds the HDF5 installation. You can check this by "
    "looking at the summary printed at the end of the cmake "
    "output.");

  /**
   * This function requires support for the MPI library.
   */
  DeclExceptionMsg(
    ExcNeedsMPI,
    "You are attempting to use functionality that is only available "
    "if deal.II was configured to use MPI."
    "\n\n"
    "You will have to ensure that your system has a usable MPI "
    "installation and re-install deal.II, making sure that cmake "
    "finds the MPI installation. You can check this by "
    "looking at the summary printed at the end of the cmake "
    "output.");

  /**
   * This function requires support for the FunctionParser library.
   */
  DeclExceptionMsg(
    ExcNeedsFunctionparser,
    "You are attempting to use functionality that is only available "
    "if deal.II was configured to use the function parser which "
    "relies on the muparser library, but cmake did not "
    "find a valid muparser library on your system and also did "
    "not choose the one that comes bundled with deal.II."
    "\n\n"
    "You will have to ensure that your system has a usable muparser "
    "installation and re-install deal.II, making sure that cmake "
    "finds the muparser installation. You can check this by "
    "looking at the summary printed at the end of the cmake "
    "output.");


  /**
   * This function requires support for the Assimp library.
   */
  DeclExceptionMsg(
    ExcNeedsAssimp,
    "You are attempting to use functionality that is only available "
    "if deal.II was configured to use Assimp, but cmake did not "
    "find a valid Assimp library."
    "\n\n"
    "You will have to ensure that your system has a usable Assimp "
    "installation and re-install deal.II, making sure that cmake "
    "finds the Assimp installation. You can check this by "
    "looking at the summary printed at the end of the cmake "
    "output.");

  /**
   * This function requires support for the Exodus II library.
   */
  DeclExceptionMsg(
    ExcNeedsExodusII,
    "You are attempting to use functionality that is only available if deal.II "
    "was configured to use Trilinos' SEACAS library (which provides ExodusII), "
    "but cmake did not find a valid SEACAS library."
    "\n\n"
    "You will have to ensure that your system has a usable ExodusII "
    "installation and re-install deal.II, making sure that cmake "
    "finds the ExodusII installation. You can check this by "
    "looking at the summary printed at the end of the cmake "
    "output.");

  /**
   * This function requires support for the CGAL library.
   */
  DeclExceptionMsg(
    ExcNeedsCGAL,
    "You are attempting to use functionality that is only available "
    "if deal.II was configured to use CGAL, but cmake did not "
    "find a valid CGAL library."
    "\n\n"
    "You will have to ensure that your system has a usable CGAL "
    "installation and re-install deal.II, making sure that cmake "
    "finds the CGAL installation. You can check this by "
    "looking at the summary printed at the end of the cmake "
    "output.");

#ifdef DEAL_II_WITH_MPI
  /**
   * Exception for MPI errors. This exception is only defined if
   * <code>deal.II</code> is compiled with MPI support. This exception should
   * be used with <code>AssertThrow</code> to check error codes of MPI
   * functions. For example:
   * @code
   * const int ierr = MPI_Isend(...);
   * AssertThrow(ierr == MPI_SUCCESS, ExcMPI(ierr));
   * @endcode
   * or, using the convenience macro <code>AssertThrowMPI</code>,
   * @code
   * const int ierr = MPI_Irecv(...);
   * AssertThrowMPI(ierr);
   * @endcode
   *
   * If the assertion fails then the error code will be used to print a helpful
   * message to the screen by utilizing the <code>MPI_Error_string</code>
   * function.
   *
   * @ingroup Exceptions
   */
  class ExcMPI : public dealii::ExceptionBase
  {
  public:
    ExcMPI(const int error_code);

    virtual void
    print_info(std::ostream &out) const override;

    const int error_code;
  };
#endif // DEAL_II_WITH_MPI



#ifdef DEAL_II_TRILINOS_WITH_SEACAS
  /**
   * Exception for ExodusII errors. This exception is only defined if
   * <code>deal.II</code> is compiled with SEACAS support, which is available
   * through Trilinos. This function should be used with the convenience macro
   * AssertThrowExodusII.
   *
   * @ingroup Exceptions
   */
  class ExcExodusII : public ExceptionBase
  {
  public:
    /**
     * Constructor.
     *
     * @param error_code The error code returned by an ExodusII function.
     */
    ExcExodusII(const int error_code);

    /**
     * Print a description of the error to the given stream.
     */
    virtual void
    print_info(std::ostream &out) const override;

    /**
     * Store the error code.
     */
    const int error_code;
  };
#endif // DEAL_II_TRILINOS_WITH_SEACAS

  /**
   * An exception to be thrown in user call-backs. See the glossary entry
   * on user call-back functions for more information.
   */
  DeclExceptionMsg(
    RecoverableUserCallbackError,
    "A user call-back function encountered a recoverable error, "
    "but the underlying library that called the call-back did not "
    "manage to recover from the error and aborted its operation."
    "\n\n"
    "See the glossary entry on user call-back functions for more "
    "information.");

  /** @} */

} /*namespace StandardExceptions*/



/**
 * In this namespace, functions in connection with the Assert and AssertThrow
 * mechanism are declared.
 *
 * @ingroup Exceptions
 */
namespace deal_II_exceptions
{
  namespace internals
  {
    /**
     * Setting this variable to false will disable deal.II's exception mechanism
     * to abort the problem. The Assert() macro will throw the exception instead
     * and the AssertNothrow() macro will just print the error message. This
     * variable should not be changed directly. Use disable_abort_on_exception()
     * instead.
     */
    extern bool allow_abort_on_exception;
  } // namespace internals

  /**
   * Set a string that is printed upon output of the message indicating a
   * triggered <tt>Assert</tt> statement. This string, which is printed in
   * addition to the usual output may indicate information that is otherwise
   * not readily available unless we are using a debugger. For example, with
   * distributed programs on cluster computers, the output of all processes is
   * redirected to the same console window. In this case, it is convenient to
   * set as additional name the name of the host on which the program runs, so
   * that one can see in which instance of the program the exception occurred.
   *
   * The string pointed to by the argument is copied, so doesn't need to be
   * stored after the call to this function.
   *
   * Previously set additional output is replaced by the argument given to
   * this function.
   *
   * @see Exceptions
   */
  void
  set_additional_assert_output(const char *const p);

  /**
   * Calling this function disables printing a stacktrace along with the other
   * output printed when an exception occurs. Most of the time, you will want
   * to see such a stacktrace; suppressing it, however, is useful if one wants
   * to compare the output of a program across different machines and systems,
   * since the stacktrace shows memory addresses and library names/paths that
   * depend on the exact setup of a machine.
   *
   * @see Exceptions
   */
  void
  suppress_stacktrace_in_exceptions();

  /**
   * Calling this function switches off the use of <tt>std::abort()</tt> when
   * an exception is created using the Assert() macro. Instead, the Exception
   * will be thrown using 'throw', so it can be caught if desired. Generally,
   * you want to abort the execution of a program when Assert() is called, but
   * it needs to be switched off if you want to log all exceptions created, or
   * if you want to test if an assertion is working correctly. This is done
   * for example in regression tests. Please note that some fatal errors will
   * still call abort(), e.g. when an exception is caught during exception
   * handling.
   *
   * @see enable_abort_on_exception
   * @see Exceptions
   */
  void
  disable_abort_on_exception();

  /**
   * Calling this function switches on the use of <tt>std::abort()</tt> when
   * an exception is created using the Assert() macro, instead of throwing it.
   * This restores the standard behavior.
   *
   * @see disable_abort_on_exception
   * @see Exceptions
   */
  void
  enable_abort_on_exception();

  /**
   * The functions in this namespace are in connection with the Assert and
   * AssertThrow mechanism but are solely for internal purposes and are not
   * for use outside the exception handling and throwing mechanism.
   *
   * @ingroup Exceptions
   */
  namespace internals
  {
    /**
     * Abort the program by printing the
     * error message provided by @p exc and calling <tt>std::abort()</tt>.
     */
    [[noreturn]] void
    abort(const ExceptionBase &exc) noexcept;

    /**
     * An enum describing how to treat an exception in issue_error_noreturn.
     */
    enum class ExceptionHandling
    {
      /**
       * Abort the program by calling <code>std::abort</code> unless
       * deal_II_exceptions::disable_abort_on_exception has been called: in
       * that case the program will throw an exception.
       */
      abort_or_throw_on_exception,
      /**
       * Throw the exception normally.
       */
      throw_on_exception
    };

    /**
     * This routine does the main work for the exception generation mechanism
     * used in the <tt>Assert</tt> and <tt>AssertThrow</tt> macros: as the
     * name implies, this function either ends by throwing an exception (if
     * @p handling is ExceptionHandling::throw_on_exception, or @p handling is try_abort_exception
     * and deal_II_exceptions::disable_abort_on_exception is false) or with a
     * call to <tt>abort</tt> (if @p handling is try_abort_exception and
     * deal_II_exceptions::disable_abort_on_exception is true).
     *
     * The actual exception object (the last argument) is typically an unnamed
     * object created in place; because we modify it, we can't take it by
     * const reference, and temporaries don't bind to non-const references.
     * So take it by value (=copy it) with a templated type to avoid slicing
     * -- the performance implications are pretty minimal anyway.
     *
     * @ref ExceptionBase
     */
    template <typename ExceptionType>
    [[noreturn]] void
    issue_error_noreturn(ExceptionHandling handling,
                         const char       *file,
                         int               line,
                         const char       *function,
                         const char       *cond,
                         const char       *exc_name,
                         ExceptionType     e)
    {
      static_assert(std::is_base_of_v<ExceptionBase, ExceptionType>,
                    "The provided exception must inherit from ExceptionBase.");
      // Fill the fields of the exception object
      e.set_fields(file, line, function, cond, exc_name);

      switch (handling)
        {
          case ExceptionHandling::abort_or_throw_on_exception:
            {
              if (dealii::deal_II_exceptions::internals::
                    allow_abort_on_exception)
                internals::abort(e);
              else
                {
                  // We are not allowed to abort, so just throw the error:
                  throw e;
                }
            }
          case ExceptionHandling::throw_on_exception:
            throw e;
          // this function should never return (and AssertNothrow can);
          // something must have gone wrong in the error handling code for us
          // to get this far, so throw an exception.
          default:
            throw ::dealii::StandardExceptions::ExcInternalError();
        }
    }

    /**
     * Internal function that does the work of issue_error_nothrow.
     */
    void
    do_issue_error_nothrow(const ExceptionBase &e) noexcept;

    /**
     * Exception generation mechanism in case we must not throw.
     *
     * @ref ExceptionBase
     *
     * @note This function is defined with a template for the same reasons as
     * issue_error_noreturn().
     */
    template <typename ExceptionType>
    void
    issue_error_nothrow(const char   *file,
                        int           line,
                        const char   *function,
                        const char   *cond,
                        const char   *exc_name,
                        ExceptionType e) noexcept
    {
      static_assert(std::is_base_of_v<ExceptionBase, ExceptionType>,
                    "The provided exception must inherit from ExceptionBase.");
      // Fill the fields of the exception object
      e.set_fields(file, line, function, cond, exc_name);
      // avoid moving a bunch of code into the header by dispatching to
      // another function:
      do_issue_error_nothrow(e);
    }
  } /*namespace internals*/

} /*namespace deal_II_exceptions*/



/**
 * A macro that serves as the main routine in the exception mechanism for debug
 * mode error checking. It asserts that a certain condition is fulfilled,
 * otherwise issues an error and aborts the program.
 *
 * A more detailed description can be found in the
 * @ref Exceptions
 * topic. It is first used in step-5 and step-6.
 * See also the <tt>ExceptionBase</tt> class for more information.
 *
 * @note Active in DEBUG mode only
 *
 * @note This and similar macro names are examples of preprocessor definitions
 * in the deal.II library that are not prefixed by a string that likely makes
 * them unique to deal.II. As a consequence, it is possible that other
 * libraries your code interfaces with define the same name, and the result
 * will be name collisions (see
 * https://en.wikipedia.org/wiki/Name_collision). One can <code>\#undef</code>
 * this macro, as well as all other macros defined by deal.II that are not
 * prefixed with either <code>DEAL</code> or <code>deal</code>, by including
 * the header <code>deal.II/base/undefine_macros.h</code> after all other
 * deal.II headers have been included.
 *
 * @ingroup Exceptions
 */
#ifdef DEBUG
#  if KOKKOS_VERSION >= 30600
#    ifdef DEAL_II_HAVE_BUILTIN_EXPECT
#      define Assert(cond, exc)                                                \
        do                                                                     \
          {                                                                    \
            KOKKOS_IF_ON_HOST(({                                               \
              if (__builtin_expect(!(cond), false))                            \
                ::dealii::deal_II_exceptions::internals::issue_error_noreturn( \
                  ::dealii::deal_II_exceptions::internals::ExceptionHandling:: \
                    abort_or_throw_on_exception,                               \
                  __FILE__,                                                    \
                  __LINE__,                                                    \
                  __PRETTY_FUNCTION__,                                         \
                  #cond,                                                       \
                  #exc,                                                        \
                  exc);                                                        \
            }))                                                                \
            KOKKOS_IF_ON_DEVICE(({                                             \
              if (!(cond))                                                     \
                Kokkos::abort(#cond);                                          \
            }))                                                                \
          }                                                                    \
        while (false)
#    else /*ifdef DEAL_II_HAVE_BUILTIN_EXPECT*/
#      define Assert(cond, exc)                                                \
        do                                                                     \
          {                                                                    \
            KOKKOS_IF_ON_HOST(({                                               \
              if (!(cond))                                                     \
                ::dealii::deal_II_exceptions::internals::issue_error_noreturn( \
                  ::dealii::deal_II_exceptions::internals::ExceptionHandling:: \
                    abort_or_throw_on_exception,                               \
                  __FILE__,                                                    \
                  __LINE__,                                                    \
                  __PRETTY_FUNCTION__,                                         \
                  #cond,                                                       \
                  #exc,                                                        \
                  exc);                                                        \
            }))                                                                \
            KOKKOS_IF_ON_DEVICE(({                                             \
              if (!(cond))                                                     \
                Kokkos::abort(#cond);                                          \
            }))                                                                \
          }                                                                    \
        while (false)
#    endif /*ifdef DEAL_II_HAVE_BUILTIN_EXPECT*/
#  else    /*if KOKKOS_VERSION >= 30600*/
#    ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST
#      ifdef DEAL_II_HAVE_BUILTIN_EXPECT
#        define Assert(cond, exc)                                              \
          do                                                                   \
            {                                                                  \
              if (__builtin_expect(!(cond), false))                            \
                ::dealii::deal_II_exceptions::internals::issue_error_noreturn( \
                  ::dealii::deal_II_exceptions::internals::ExceptionHandling:: \
                    abort_or_throw_on_exception,                               \
                  __FILE__,                                                    \
                  __LINE__,                                                    \
                  __PRETTY_FUNCTION__,                                         \
                  #cond,                                                       \
                  #exc,                                                        \
                  exc);                                                        \
            }                                                                  \
          while (false)
#      else /*ifdef DEAL_II_HAVE_BUILTIN_EXPECT*/
#        define Assert(cond, exc)                                              \
          do                                                                   \
            {                                                                  \
              if (!(cond))                                                     \
                ::dealii::deal_II_exceptions::internals::issue_error_noreturn( \
                  ::dealii::deal_II_exceptions::internals::ExceptionHandling:: \
                    abort_or_throw_on_exception,                               \
                  __FILE__,                                                    \
                  __LINE__,                                                    \
                  __PRETTY_FUNCTION__,                                         \
                  #cond,                                                       \
                  #exc,                                                        \
                  exc);                                                        \
            }                                                                  \
          while (false)
#      endif /*ifdef DEAL_II_HAVE_BUILTIN_EXPECT*/
#    else    /*#ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST*/
#      define Assert(cond, exc)     \
        do                          \
          {                         \
            if (!(cond))            \
              Kokkos::abort(#cond); \
          }                         \
        while (false)
#    endif /*ifdef KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST*/
#  endif   /*KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST*/
#else      /*ifdef DEBUG*/
#  ifdef DEAL_II_HAVE_CXX20
/*
 * In order to avoid unused parameters (etc.) warnings we need to use cond
 * and exc without actually evaluating the expression and generating code.
 * We accomplish this by using decltype(...) and create a dummy pointer
 * with these signatures. Notably, this approach works with C++20 onwards.
 */
#    define Assert(cond, exc)                                  \
      do                                                       \
        {                                                      \
          typename std::remove_reference<decltype(cond)>::type \
            *dealii_assert_variable_a = nullptr;               \
          typename std::remove_reference<decltype(exc)>::type  \
            *dealii_assert_variable_b = nullptr;               \
          (void)dealii_assert_variable_a;                      \
          (void)dealii_assert_variable_b;                      \
        }                                                      \
      while (false)
#  else
#    define Assert(cond, exc) \
      do                      \
        {                     \
        }                     \
      while (false)
#  endif
#endif /*ifdef DEBUG*/



/**
 * A variant of the <tt>Assert</tt> macro above that exhibits the same runtime
 * behavior as long as disable_abort_on_exception was not called.
 *
 * However, if disable_abort_on_exception was called, this macro merely prints
 * the exception that would be thrown to deallog and continues normally
 * without throwing an exception.
 *
 * A more detailed description can be found in the
 * @ref Exceptions
 * topic, in the discussion about the corner case at the bottom of the page.
 *
 * @note This and similar macro names are examples of preprocessor definitions
 * in the deal.II library that are not prefixed by a string that likely makes
 * them unique to deal.II. As a consequence, it is possible that other
 * libraries your code interfaces with define the same name, and the result
 * will be name collisions (see
 * https://en.wikipedia.org/wiki/Name_collision). One can <code>\#undef</code>
 * this macro, as well as all other macros defined by deal.II that are not
 * prefixed with either <code>DEAL</code> or <code>deal</code>, by including
 * the header <code>deal.II/base/undefine_macros.h</code> after all other
 * deal.II headers have been included.
 *
 * @note Active in DEBUG mode only
 * @ingroup Exceptions
 */
#ifdef DEBUG
#  ifdef DEAL_II_HAVE_BUILTIN_EXPECT
#    define AssertNothrow(cond, exc)                                      \
      do                                                                  \
        {                                                                 \
          if (__builtin_expect(!(cond), false))                           \
            ::dealii::deal_II_exceptions::internals::issue_error_nothrow( \
              __FILE__, __LINE__, __PRETTY_FUNCTION__, #cond, #exc, exc); \
        }                                                                 \
      while (false)
#  else /*ifdef DEAL_II_HAVE_BUILTIN_EXPECT*/
#    define AssertNothrow(cond, exc)                                      \
      do                                                                  \
        {                                                                 \
          if (!(cond))                                                    \
            ::dealii::deal_II_exceptions::internals::issue_error_nothrow( \
              __FILE__, __LINE__, __PRETTY_FUNCTION__, #cond, #exc, exc); \
        }                                                                 \
      while (false)
#  endif /*ifdef DEAL_II_HAVE_BUILTIN_EXPECT*/
#else
#  define AssertNothrow(cond, exc) \
    do                             \
      {                            \
      }                            \
    while (false)
#endif

/**
 * A macro that serves as the main routine in the exception mechanism for
 * dynamic error checking. It asserts that a certain condition is fulfilled,
 * otherwise
 * throws an exception via the C++ @p throw mechanism. This exception can
 * be caught via a @p catch clause, as is shown in step-6 and all following
 * tutorial programs.
 *
 * A more detailed description can be found in the
 * @ref Exceptions
 * topic. It is first used in step-9 and step-13.
 * See also the <tt>ExceptionBase</tt> class for more information.
 *
 * @note This and similar macro names are examples of preprocessor definitions
 * in the deal.II library that are not prefixed by a string that likely makes
 * them unique to deal.II. As a consequence, it is possible that other
 * libraries your code interfaces with define the same name, and the result
 * will be name collisions (see
 * https://en.wikipedia.org/wiki/Name_collision). One can <code>\#undef</code>
 * this macro, as well as all other macros defined by deal.II that are not
 * prefixed with either <code>DEAL</code> or <code>deal</code>, by including
 * the header <code>deal.II/base/undefine_macros.h</code> after all other
 * deal.II headers have been included.
 *
 * @note Active in both DEBUG and RELEASE modes
 * @ingroup Exceptions
 */
#ifdef DEAL_II_HAVE_BUILTIN_EXPECT
#  define AssertThrow(cond, exc)                                         \
    do                                                                   \
      {                                                                  \
        if (__builtin_expect(!(cond), false))                            \
          ::dealii::deal_II_exceptions::internals::issue_error_noreturn( \
            ::dealii::deal_II_exceptions::internals::ExceptionHandling:: \
              throw_on_exception,                                        \
            __FILE__,                                                    \
            __LINE__,                                                    \
            __PRETTY_FUNCTION__,                                         \
            #cond,                                                       \
            #exc,                                                        \
            exc);                                                        \
      }                                                                  \
    while (false)
#else /*ifdef DEAL_II_HAVE_BUILTIN_EXPECT*/
#  define AssertThrow(cond, exc)                                         \
    do                                                                   \
      {                                                                  \
        if (!(cond))                                                     \
          ::dealii::deal_II_exceptions::internals::issue_error_noreturn( \
            ::dealii::deal_II_exceptions::internals::ExceptionHandling:: \
              throw_on_exception,                                        \
            __FILE__,                                                    \
            __LINE__,                                                    \
            __PRETTY_FUNCTION__,                                         \
            #cond,                                                       \
            #exc,                                                        \
            exc);                                                        \
      }                                                                  \
    while (false)
#endif /*ifdef DEAL_II_HAVE_BUILTIN_EXPECT*/


/**
 * `DEAL_II_NOT_IMPLEMENTED` is a macro (that looks like a function call
 * when used as in `DEAL_II_NOT_IMPLEMENTED();`) that is used to raise an
 * error in places where a piece of code is not yet implemented. If a code
 * runs into such a place, it will be aborted with an error message that
 * explains the situation, along with a backtrace of how the code ended
 * up in this place. Alternatively, if
 * deal_II_exceptions::internals::ExceptionHandling::abort_or_throw_on_exception
 * is set to ExceptionHandling::throw_on_exception, then the corresponding
 * error is thrown as a C++ exception that can be caught (though in
 * many cases codes will then find it difficult to do what they wanted
 * to do).
 *
 * This macro is first used in step-8 of the tutorial.
 *
 * A typical case where it is used would look as follows: Assume that we want
 * to implement a function that describes the right hand side of an equation
 * that corresponds to a known solution (i.e., we want to use the "Method
 * of manufactured solutions", see step-7). We have computed the right
 * hand side that corresponds to the 1d and 2d solutions, but we have been
 * too lazy so far to do the calculations for the 3d case, perhaps because
 * we first want to test correctness in 1d and 2d before moving on to the 3d
 * case. We could then write this right hand side as follows (the specific
 * formulas in the `return` statements are not important):
 * @code
 *   template <int dim>
 *   double right_hand_side (const Point<dim> &x)
 *   {
 *     if (dim==1)
 *       return x[0]*std::sin(x[0]);
 *     else if (dim==2)
 *       return x[0]*std::sin(x[0])*std::sin(x[1];
 *     else
 *       DEAL_II_NOT_IMPLEMENTED();
 *   }
 * @endcode
 * Here, the call to `DEAL_II_NOT_IMPLEMENTED()` simply indicates that we
 * haven't gotten around to filling in this code block. If someone ends up
 * running the program in 3d, execution will abort in the location with an
 * error message that indicates where this happened and why.
 */
#define DEAL_II_NOT_IMPLEMENTED()                                \
  ::dealii::deal_II_exceptions::internals::issue_error_noreturn( \
    ::dealii::deal_II_exceptions::internals::ExceptionHandling:: \
      abort_or_throw_on_exception,                               \
    __FILE__,                                                    \
    __LINE__,                                                    \
    __PRETTY_FUNCTION__,                                         \
    nullptr,                                                     \
    nullptr,                                                     \
    ::dealii::StandardExceptions::ExcNotImplemented())


/**
 * `DEAL_II_ASSERT_UNREACHABLE` is a macro (that looks like a function call
 * when used as in `DEAL_II_ASSERT_UNREACHABLE();`) that is used to raise an
 * error in places where the programmer believed that execution should
 * never get to. If a code
 * runs into such a place, it will be aborted with an error message that
 * explains the situation, along with a backtrace of how the code ended
 * up in this place. Alternatively, if
 * deal_II_exceptions::internals::ExceptionHandling::abort_or_throw_on_exception
 * is set to ExceptionHandling::throw_on_exception, then the corresponding
 * error is thrown as a C++ exception that can be caught (though in
 * many cases codes will then find it difficult to do what they wanted
 * to do).
 *
 * A typical case where it is used would look as follows. In many cases,
 * one has a finite enumeration of things that can happen, and one runs
 * through those in a sequence of `if`-`else` blocks, or perhaps
 * with a `switch` selection and a number of `case` statements. Of
 * course, if the code is correct, if all possible cases are handled,
 * nothing terrible can happen -- though perhaps it is worth making sure
 * that we have really covered all cases by using `DEAL_II_ASSERT_UNREACHABLE()`
 * as the *last* case. Here is an example:
 * @code
 *   enum OutputFormat { vtk, vtu };
 *
 *   void write_output (const OutputFormat format)
 *   {
 *     if (format == vtk)
 *       {
 *         ... write in VTK format ...
 *       }
 *     else // must not clearly be VTU format
 *       {
 *         ... write in VTU format ...
 *       }
 *   }
 * @endcode
 * The issue here is "Are we really sure it is VTU format if we end up in
 * the `else` block"? There are two reasons that should make us suspicious.
 * First, the authors of the code may later have expanded the list of options
 * in the `OutputFormat` enum, but forgotten to also update the
 * `write_output()` function. We may then end up in the `else` branch even
 * though the argument indicates the now possible third option that was added
 * to `OutputFormat`. The second possibility to consider is that enums are
 * really just fancy ways of using integers; from a language perspective, it
 * is allowed to pass *any* integer to `write_output()`, even values that do
 * not match either `vtk` or `vtu`. This is then clearly a bug in the program,
 * but one that we are better off if we catch it as early as possible.
 *
 * We can guard against both cases by writing the code as follows instead:
 * @code
 *   enum OutputFormat { vtk, vtu };
 *
 *   void write_output (const OutputFormat format)
 *   {
 *     if (format == vtk)
 *       {
 *         ... write in VTK format ...
 *       }
 *     else if (format == vtu)
 *       {
 *         ... write in VTU format ...
 *       }
 *     else // we shouldn't get here, but if we did, abort the program now!
 *       DEAL_II_ASSERT_UNREACHABLE();
 *   }
 * @endcode
 *
 * This macro is first used in step-7, where we show another example of
 * a context where it is frequently used.
 */
#define DEAL_II_ASSERT_UNREACHABLE()                             \
  ::dealii::deal_II_exceptions::internals::issue_error_noreturn( \
    ::dealii::deal_II_exceptions::internals::ExceptionHandling:: \
      abort_or_throw_on_exception,                               \
    __FILE__,                                                    \
    __LINE__,                                                    \
    __PRETTY_FUNCTION__,                                         \
    nullptr,                                                     \
    nullptr,                                                     \
    ::dealii::StandardExceptions::ExcMessage(                    \
      "The program has hit a line of code that the programmer "  \
      "marked with the macro DEAL_II_ASSERT_UNREACHABLE() to "   \
      "indicate that the program should never reach this "       \
      "location. You will have to find out (best done in a "     \
      "debugger) why that happened. Typical reasons include "    \
      "passing invalid arguments to functions (for example, if " \
      "a function takes an 'enum' with two possible values "     \
      "as argument, but you call the function with a third "     \
      "value), or if the programmer of the code that triggered " \
      "the error believed that a variable can only have "        \
      "specific values, but either that assumption is wrong "    \
      "or the computation of that value is buggy."               \
      "\n\n"                                                     \
      "In those latter conditions, where some internal "         \
      "assumption is not satisfied, there may not be very "      \
      "much you can do if you encounter such an exception, "     \
      "since it indicates an error in deal.II, not in your "     \
      "own program. If that is the situation you encounter, "    \
      "try to come up with "                                     \
      "the smallest possible program that still demonstrates "   \
      "the error and contact the deal.II mailing lists with it " \
      "to obtain help."))


namespace deal_II_exceptions
{
  namespace internals
  {
    /**
     * A function that compares two values for equality, after converting to a
     * common type to avoid compiler warnings when comparing objects of
     * different types (e.g., unsigned and signed variables).
     */
    template <typename T, typename U>
    inline DEAL_II_HOST_DEVICE constexpr bool
    compare_for_equality(const T &t, const U &u)
    {
      using common_type = std::common_type_t<T, U>;
      return static_cast<common_type>(t) == static_cast<common_type>(u);
    }


    /**
     * A function that compares two values with `operator<`, after converting to
     * a common type to avoid compiler warnings when comparing objects of
     * different types (e.g., unsigned and signed variables).
     */
    template <typename T, typename U>
    inline DEAL_II_HOST_DEVICE constexpr bool
    compare_less_than(const T &t, const U &u)
    {
      using common_type = std::common_type_t<T, U>;
      return (static_cast<common_type>(t) < static_cast<common_type>(u));
    }
  } // namespace internals
} // namespace deal_II_exceptions


/**
 * Special assertion for dimension mismatch.
 *
 * Since this is used very often and always repeats the arguments, we
 * introduce this special assertion for ExcDimensionMismatch in order to keep
 * the user codes shorter.
 *
 * @note This and similar macro names are examples of preprocessor definitions
 * in the deal.II library that are not prefixed by a string that likely makes
 * them unique to deal.II. As a consequence, it is possible that other
 * libraries your code interfaces with define the same name, and the result
 * will be name collisions (see
 * https://en.wikipedia.org/wiki/Name_collision). One can <code>\#undef</code>
 * this macro, as well as all other macros defined by deal.II that are not
 * prefixed with either <code>DEAL</code> or <code>deal</code>, by including
 * the header <code>deal.II/base/undefine_macros.h</code> after all other
 * deal.II headers have been included.
 *
 * @ingroup Exceptions
 */
#define AssertDimension(dim1, dim2)                                           \
  Assert(::dealii::deal_II_exceptions::internals::compare_for_equality(dim1,  \
                                                                       dim2), \
         dealii::ExcDimensionMismatch((dim1), (dim2)))

/**
 * Special assertion for integer conversions.
 *
 * deal.II does not always use the same integer types as its dependencies. For
 * example, PETSc uses signed integers whereas deal.II uses unsigned integers.
 * This assertion checks that we can successfully convert between two index
 * types.
 */
#define AssertIntegerConversion(index1, index2)                         \
  Assert(::dealii::deal_II_exceptions::internals::compare_for_equality( \
           index1, index2),                                             \
         dealii::ExcInvalidIntegerConversion((index1), (index2)))

/**
 * Special assertion for integer conversions which will throw exceptions.
 * Otherwise this is the same as AssertIntegerConversion.
 */
#define AssertThrowIntegerConversion(index1, index2)                         \
  AssertThrow(::dealii::deal_II_exceptions::internals::compare_for_equality( \
                index1, index2),                                             \
              dealii::ExcInvalidIntegerConversion((index1), (index2)))

/**
 * An assertion that tests whether <tt>vec</tt> has size <tt>dim1</tt>, and
 * each entry of the vector is itself an array that has the size <tt>dim2</tt>.
 *
 * @note This and similar macro names are examples of preprocessor definitions
 * in the deal.II library that are not prefixed by a string that likely makes
 * them unique to deal.II. As a consequence, it is possible that other
 * libraries your code interfaces with define the same name, and the result
 * will be name collisions (see
 * https://en.wikipedia.org/wiki/Name_collision). One can <code>\#undef</code>
 * this macro, as well as all other macros defined by deal.II that are not
 * prefixed with either <code>DEAL</code> or <code>deal</code>, by including
 * the header <code>deal.II/base/undefine_macros.h</code> after all other
 * deal.II headers have been included.
 *
 * @ingroup Exceptions
 */
#define AssertVectorVectorDimension(VEC, DIM1, DIM2) \
  AssertDimension(VEC.size(), DIM1);                 \
  for (const auto &subvector : VEC)                  \
    {                                                \
      (void)subvector;                               \
      AssertDimension(subvector.size(), DIM2);       \
    }

namespace internal
{
  // Workaround to allow for commas in template parameter lists
  // in preprocessor macros as found in
  // https://stackoverflow.com/questions/13842468/comma-in-c-c-macro
  template <typename F>
  struct argument_type;

  template <typename T, typename U>
  struct argument_type<T(U)>
  {
    using type = U;
  };

  template <typename F>
  using argument_type_t = typename argument_type<F>::type;
} // namespace internal

/**
 * An assertion that tests that a given index is within the half-open
 * range <code>[0,range)</code>. It throws an exception object
 * <code>ExcIndexRange(index,0,range)</code> if the assertion
 * fails.
 *
 * @note This and similar macro names are examples of preprocessor definitions
 * in the deal.II library that are not prefixed by a string that likely makes
 * them unique to deal.II. As a consequence, it is possible that other
 * libraries your code interfaces with define the same name, and the result
 * will be name collisions (see
 * https://en.wikipedia.org/wiki/Name_collision). One can <code>\#undef</code>
 * this macro, as well as all other macros defined by deal.II that are not
 * prefixed with either <code>DEAL</code> or <code>deal</code>, by including
 * the header <code>deal.II/base/undefine_macros.h</code> after all other
 * deal.II headers have been included.
 *
 * @ingroup Exceptions
 */
#define AssertIndexRange(index, range)                                       \
  Assert(::dealii::deal_II_exceptions::internals::compare_less_than(index,   \
                                                                    range),  \
         dealii::ExcIndexRangeType<::dealii::internal::argument_type_t<void( \
           std::common_type_t<decltype(index), decltype(range)>)>>((index),  \
                                                                   0,        \
                                                                   (range)))

/**
 * An assertion that checks whether a number is finite or not. We explicitly
 * cast the number to std::complex to match the signature of the exception
 * (see there for an explanation of why we use std::complex at all) and to
 * satisfy the fact that std::complex has no implicit conversions.
 *
 * @note This and similar macro names are examples of preprocessor definitions
 * in the deal.II library that are not prefixed by a string that likely makes
 * them unique to deal.II. As a consequence, it is possible that other
 * libraries your code interfaces with define the same name, and the result
 * will be name collisions (see
 * https://en.wikipedia.org/wiki/Name_collision). One can <code>\#undef</code>
 * this macro, as well as all other macros defined by deal.II that are not
 * prefixed with either <code>DEAL</code> or <code>deal</code>, by including
 * the header <code>deal.II/base/undefine_macros.h</code> after all other
 * deal.II headers have been included.
 *
 * @ingroup Exceptions
 */
#define AssertIsFinite(number)               \
  Assert(dealii::numbers::is_finite(number), \
         dealii::ExcNumberNotFinite(std::complex<double>(number)))

/**
 * Assert that a geometric object is not used. This assertion is used when
 * constructing triangulations and should normally not be used inside user
 * codes.
 */
#define AssertIsNotUsed(obj) Assert((obj)->used() == false, ExcInternalError())

#ifdef DEAL_II_WITH_MPI
/**
 * An assertion that checks whether or not an error code returned by an MPI
 * function is equal to <code>MPI_SUCCESS</code>. If the check fails then an
 * exception of type ExcMPI is thrown with the given error code as an
 * argument.
 *
 * @note This and similar macro names are examples of preprocessor definitions
 * in the deal.II library that are not prefixed by a string that likely makes
 * them unique to deal.II. As a consequence, it is possible that other
 * libraries your code interfaces with define the same name, and the result
 * will be name collisions (see
 * https://en.wikipedia.org/wiki/Name_collision). One can <code>\#undef</code>
 * this macro, as well as all other macros defined by deal.II that are not
 * prefixed with either <code>DEAL</code> or <code>deal</code>, by including
 * the header <code>deal.II/base/undefine_macros.h</code> after all other
 * deal.II headers have been included.
 *
 * @note Active only if deal.II is compiled with MPI
 * @ingroup Exceptions
 */
#  define AssertThrowMPI(error_code) \
    AssertThrow(error_code == MPI_SUCCESS, dealii::ExcMPI(error_code))
#else
#  define AssertThrowMPI(error_code) \
    do                               \
      {                              \
      }                              \
    while (false)
#endif // DEAL_II_WITH_MPI

#ifdef DEAL_II_TRILINOS_WITH_SEACAS
/**
 * Assertion that checks that the error code produced by calling an ExodusII
 * routine is equal to EX_NOERR (which is zero).
 *
 * @note This and similar macro names are examples of preprocessor definitions
 * in the deal.II library that are not prefixed by a string that likely makes
 * them unique to deal.II. As a consequence, it is possible that other
 * libraries your code interfaces with define the same name, and the result
 * will be name collisions (see
 * https://en.wikipedia.org/wiki/Name_collision). One can <code>\#undef</code>
 * this macro, as well as all other macros defined by deal.II that are not
 * prefixed with either <code>DEAL</code> or <code>deal</code>, by including
 * the header <code>deal.II/base/undefine_macros.h</code> after all other
 * deal.II headers have been included.
 *
 * @ingroup Exceptions
 */
#  define AssertThrowExodusII(error_code) \
    AssertThrow(error_code == 0,          \
                dealii::StandardExceptions::ExcExodusII(error_code));
#endif // DEAL_II_TRILINOS_WITH_SEACAS

using namespace StandardExceptions;

DEAL_II_NAMESPACE_CLOSE

#endif
