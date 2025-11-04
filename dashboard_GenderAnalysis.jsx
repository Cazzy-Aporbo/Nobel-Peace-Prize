import React, { useMemo } from 'react'
import styled from 'styled-components'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { theme } from '../../theme'

const Section = styled.section`
  margin-bottom: ${theme.spacing.xxl};
`

const SectionTitle = styled.h2`
  color: ${theme.colors.mint};
  font-size: 2rem;
  margin-bottom: ${theme.spacing.lg};
  padding-bottom: ${theme.spacing.md};
  border-bottom: 2px solid ${theme.colors.teal};
`

const ChartContainer = styled.div`
  background: rgba(71, 105, 117, 0.1);
  padding: ${theme.spacing.xl};
  border-radius: ${theme.borderRadius.md};
  margin: ${theme.spacing.xl} 0;
  border: 2px solid ${theme.colors.teal};
`

const ChartTitle = styled.h3`
  color: ${theme.colors.purple};
  font-size: 1.5rem;
  margin-bottom: ${theme.spacing.lg};
`

const CriticalFinding = styled.div`
  background: linear-gradient(135deg, rgba(87, 82, 196, 0.3), rgba(71, 105, 117, 0.3));
  padding: ${theme.spacing.xl};
  border-radius: ${theme.borderRadius.md};
  margin: ${theme.spacing.xl} 0;
  border-left: 5px solid ${theme.colors.purple};
`

const FindingText = styled.p`
  color: ${theme.colors.light};
  line-height: 1.8;
  font-size: 1.1rem;
  margin-bottom: ${theme.spacing.md};
`

const GenderAnalysis = ({ data }) => {
  const categoryGenderData = useMemo(() => {
    if (!data) return []
    
    const categories = {}
    data.forEach(d => {
      if (!categories[d.category]) {
        categories[d.category] = { category: d.category, female: 0, male: 0 }
      }
      if (d.sex === 'female') categories[d.category].female++
      if (d.sex === 'male') categories[d.category].male++
    })
    
    return Object.values(categories).map(c => ({
      category: c.category,
      femalePercent: ((c.female / (c.female + c.male)) * 100).toFixed(2),
      malePercent: ((c.male / (c.female + c.male)) * 100).toFixed(2)
    }))
  }, [data])

  if (!data) return null

  return (
    <Section>
      <SectionTitle>Gender Disparity Analysis</SectionTitle>
      
      <CriticalFinding>
        <FindingText>
          <strong style={{ color: theme.colors.mint, fontSize: '1.3rem' }}>Critical Finding: Systematic Underrepresentation</strong>
        </FindingText>
        <FindingText>
          Despite 125 years of Nobel Prize history, female representation remains at 6.52%. This systematic 
          underrepresentation persists across all categories, with Physics (2.10%) and Economic Sciences (1.89%) 
          showing the most severe disparities. This pattern indicates structural barriers rather than incidental 
          bias, requiring immediate institutional intervention.
        </FindingText>
      </CriticalFinding>

      <ChartContainer>
        <ChartTitle>Female Representation by Category</ChartTitle>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={categoryGenderData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke={theme.colors.teal} opacity={0.3} />
            <XAxis type="number" stroke={theme.colors.light} domain={[0, 100]} />
            <YAxis type="category" dataKey="category" stroke={theme.colors.light} width={150} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: theme.colors.dark, 
                border: `2px solid ${theme.colors.teal}`,
                borderRadius: theme.borderRadius.sm,
                color: theme.colors.light
              }} 
            />
            <Legend />
            <Bar dataKey="femalePercent" fill={theme.colors.purple} name="Female %" />
            <Bar dataKey="malePercent" fill={theme.colors.teal} name="Male %" />
          </BarChart>
        </ResponsiveContainer>
      </ChartContainer>

      <CriticalFinding>
        <FindingText>
          <strong style={{ color: theme.colors.mint }}>Healthcare AI Parallel:</strong> The systematic gender bias 
          revealed in Nobel Prize awards parallels bias patterns observed in medical AI systems. Historical exclusion 
          of women from clinical trials creates training data bias, leading to algorithms that underperform for 
          female patients. Both phenomena require systematic intervention through frameworks like R.O.A.D. (Representation, 
          Outcome, Algorithmic, Distributional) and ADAPT (Assess, Design, Audit, Participate, Trace).
        </FindingText>
      </CriticalFinding>
    </Section>
  )
}

export default GenderAnalysis
